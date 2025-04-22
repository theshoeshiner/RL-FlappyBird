package com.kingyu.rlbird.ai;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.rl.agent.EpsilonGreedy;
import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.LinearTracker;
import ai.djl.training.tracker.Tracker;
import com.kingyu.rlbird.game.FlappyBird;
import com.kingyu.rlbird.rl.agent.QAgent;
import com.kingyu.rlbird.util.Constant;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.nio.file.Paths;

public class FlappyBirdTrainer {

    private static final Logger logger = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    public static int SCREEN_SIZE = 80;
    public static final int SAVE_EVERY_STEPS = 100; // save model every 1,000 epochs

    private final int inputFrames;
    private final int batchSize;
    private final int bufferSize;
    private final float startEpsilon;
    private final float endEpsilon;
    private final float rewardDiscount;
    private final boolean preTrained;
    private final boolean withGraphics;
    private final boolean test;
    private final String modelNamePrefix;

    public FlappyBirdTrainer(int inputFrames, int batchSize, int bufferSize, float startEpsilon, float endEpsilon, float rewardDiscount, boolean preTrained, boolean withGraphics, boolean test) {
        this.inputFrames = inputFrames;
        this.batchSize = batchSize;
        this.bufferSize = bufferSize;
        this.startEpsilon = startEpsilon;
        this.endEpsilon = endEpsilon;
        this.rewardDiscount = rewardDiscount;
        this.preTrained = preTrained;
        this.withGraphics = withGraphics;
        this.test = test;
        modelNamePrefix = "dqn-replay_" +bufferSize+"-batch_"+batchSize;
    }

    public void run() throws ParseException, IOException, MalformedModelException {

            Model model = createOrLoadModel();
            if (test) {
                test(model);
            } else {
                train(model);
            }


    }

    private Model createOrLoadModel() throws MalformedModelException {
        Model model = Model.newInstance("QNetwork");

        model.setBlock(getBlock());
        if (preTrained) {
            //logger.info("loading model: {}",modelNamePrefix);
            try {
                model.load(Paths.get(Constant.MODEL_PATH), modelNamePrefix);
                logger.info("loaded model at epoch {}",model.getProperty("Epoch"));
            } catch (IOException ignored) {
            }


        }
        return model;
    }

    private void train(Model model) {

        logger.info("rewardDiscount: {}",rewardDiscount);
        logger.info("startEpsilon: {}",startEpsilon);
        logger.info("endEpsilon: {}",endEpsilon);
        logger.info("bufferSize: {}",bufferSize);
        logger.info("batchSize: {}",batchSize);


        NDManager gameManager = NDManager.newBaseManager();

        FlappyBird game = new FlappyBird(gameManager, batchSize, bufferSize, withGraphics);

        DefaultTrainingConfig config = setupTrainingConfig();
        Trainer trainer = model.newTrainer(config);

        trainer.initialize(new Shape(batchSize, inputFrames, SCREEN_SIZE, SCREEN_SIZE));

        RlAgent agent = new QAgent(trainer, rewardDiscount);
        Tracker exploreRate =
                LinearTracker.builder()
                        .setBaseValue(startEpsilon)
                        .optSlope(-(startEpsilon - endEpsilon) / TrainerThread.TRAIN_STEPS)
                        .optMinValue(endEpsilon)
                        .build();
        agent = new EpsilonGreedy(agent, exploreRate);

        TrainerThread trainerThread = new TrainerThread(model,agent,trainer,game);

        GameThread gameThread = new GameThread(agent,trainer,model,game, trainerThread);

        gameThread.start();



        while(gameThread.isAlive()) {
            try {
                Thread.sleep(10000);
                //logger.info("still waiting...");
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        logger.info("Game thread was done, ending main thread");


    }

    private void test(Model model) {
        FlappyBird game = new FlappyBird(NDManager.newBaseManager(), 1, 1, true);
        DefaultTrainingConfig config = setupTrainingConfig();
        try (Trainer trainer = model.newTrainer(config)) {
            RlAgent agent = new QAgent(trainer, rewardDiscount);
            while (true) {
                game.runEnvironment(agent, false);
            }
        }
    }


    private static SequentialBlock getBlock() {
        // conv -> conv -> conv -> fc -> fc
        return new SequentialBlock()
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(8, 8))
                        .optStride(new Shape(4, 4))
                        .optPadding(new Shape(3, 3))
                        .setFilters(4).build())
                .add(Activation::relu)

                .add(Conv2d.builder()
                        .setKernelShape(new Shape(4, 4))
                        .optStride(new Shape(2, 2))
                        .setFilters(32).build())
                .add(Activation::relu)

                .add(Conv2d.builder()
                        .setKernelShape(new Shape(3, 3))
                        .optStride(new Shape(1, 1))
                        .setFilters(64).build())
                .add(Activation::relu)

                .add(Blocks.batchFlattenBlock())
                .add(Linear
                        .builder()
                        .setUnits(512).build())
                .add(Activation::relu)

                .add(Linear
                        .builder()
                        .setUnits(2).build());
    }

    private DefaultTrainingConfig setupTrainingConfig() {
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                .addTrainingListeners(TrainingListener.Defaults.basic())

                .optOptimizer(Adam.builder().optLearningRateTracker(Tracker.fixed(1e-6f)).build())
                .addEvaluator(new Accuracy())
                .optInitializer(new NormalInitializer(), Parameter.Type.WEIGHT)
                ;
        SaveModelTrainingListener saveModelTrainingListener = new SaveModelTrainingListener(Paths.get(Constant.MODEL_PATH).toString(),modelNamePrefix,1);
        saveModelTrainingListener.setSaveModelCallback(trainer -> {
            logger.info("saving model with epoch: {}",trainer.getModel().getProperty("Epoch"));
        });
        config.addTrainingListeners(saveModelTrainingListener);
        return config;
    }
}
