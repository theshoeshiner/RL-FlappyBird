package com.kingyu.rlbird.ai;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;
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
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.LinearTracker;
import ai.djl.training.tracker.Tracker;
import com.kingyu.rlbird.game.FlappyBird;
import com.kingyu.rlbird.rl.agent.QAgent;
import com.kingyu.rlbird.util.Arguments;
import com.kingyu.rlbird.util.Constant;
import org.apache.commons.cli.ParseException;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math.util.MathUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Paths;

public final class TrainBird {

    private static final Logger logger = LoggerFactory.getLogger(TrainBird.class);

    public static final int EXPLORE = 3_000_000; // frames over which to anneal epsilon og was 3_000_000
    public static final int SAVE_EVERY_STEPS = 100_000; // save model every 100,000 step
    public static final int REPLAY_BUFFER_SIZE = 10_000; // number of previous transitions to remember TODO DCW Org version was 50k
    public static final float REWARD_DISCOUNT = 0.9f; // decay rate of past observations
    public static final String PARAMS_PREFIX = "dqn-trained";

    public static final float INITIAL_EPSILON = 0.01f; //0.01
    public static final float FINAL_EPSILON = 0.0001f; // 0.0001


    public static int INPUT_FRAMES = 4;
    public static int SCREEN_SIZE = 80;

    private TrainBird() {}

    public static void main(String[] args) throws ParseException, IOException, MalformedModelException {

        System.out.println(Device.cpu());
        System.out.println(Device.gpu());
        System.out.println(Device.gpu(1));

        System.out.println("GPU count: " + Engine.getInstance().getGpuCount());
        Device d = Device.gpu(1);

        try {

            Arguments arguments = Arguments.parseArgs(args);
            Model model = createOrLoadModel(arguments);
            if (arguments.isTesting()) {
                test(model);
            } else {
                train(arguments, model);
            }

        }
        catch(Exception e){
            e.printStackTrace();
        }
    }

    public static Model createOrLoadModel(Arguments arguments) throws IOException, MalformedModelException {
        Model model = Model.newInstance("QNetwork");
        model.setBlock(getBlock());
        if (arguments.usePreTrained()) {
            model.load(Paths.get(Constant.MODEL_PATH), PARAMS_PREFIX);
        }
        return model;
    }

    public static void train(Arguments arguments, Model model) {
        boolean withGraphics = arguments.withGraphics();
        int batchSize = arguments.getBatchSize();  // size of mini batch
        long start = System.currentTimeMillis();

        NDManager gameManager = NDManager.newBaseManager();

        FlappyBird game = new FlappyBird(gameManager, batchSize, REPLAY_BUFFER_SIZE, withGraphics);


        DefaultTrainingConfig config = setupTrainingConfig();
        Trainer trainer = model.newTrainer(config);

        trainer.initialize(new Shape(batchSize, INPUT_FRAMES, 80, 80));

       /* trainer.initialize(
                new Shape(batchSize, INPUT_FRAMES, SCREEN_SIZE, SCREEN_SIZE), // state in,
                //new Shape(batchSize), // action space?
                new Shape(batchSize, 2) // action - flap or dont

        );*/
        trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));

        RlAgent agent = new QAgent(trainer, REWARD_DISCOUNT);
        Tracker exploreRate =
                LinearTracker.builder()
                        .setBaseValue(INITIAL_EPSILON)
                        .optSlope(-(INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE)
                        .optMinValue(FINAL_EPSILON)
                        .build();
        agent = new EpsilonGreedy(agent, exploreRate);

        int reportMod = 100;
        int scale = 3;

        DescriptiveStatistics score = new DescriptiveStatistics(reportMod);
        DescriptiveStatistics steps = new DescriptiveStatistics(reportMod);
        DescriptiveStatistics scoreTotal = new DescriptiveStatistics();
        DescriptiveStatistics stepsTotal = new DescriptiveStatistics();

        long totalSteps = 0;
        BirdTrainer birdTrainer = new BirdTrainer(agent,trainer,game);
        boolean replayBufferFull = false;
        int games = 0;

        training: while(true) {

            int batchSteps = 0;

            // Make sure we generate at least enough steps to cover the last processed batch
            while(batchSteps < batchSize) {

                float result = game.runEnvironment(agent, true); // runs game until a terminal step, training is assumed to be true
                int gameSteps = game.getGameStep();
                batchSteps += gameSteps;

                score.addValue(game.getScore());
                steps.addValue(game.gameStep);
                scoreTotal.addValue(game.getScore());
                stepsTotal.addValue(game.gameStep);

                totalSteps += game.gameStep;
                if (!replayBufferFull && totalSteps > REPLAY_BUFFER_SIZE) {
                    replayBufferFull = true;
                    birdTrainer.start();

                }

                if (games % reportMod == 0) {
                    long dur = System.currentTimeMillis() - start;

                    logger.info("GAME: {} TOTAL-STEPS: {} ", games, totalSteps);
                    logger.info("RATE: {} steps/s {} games/s", (totalSteps * 1000L) / dur,(games * 1000L) / dur);

                    logger.info("ARRAYS {}", gameManager.getManagedArrays().size());
                    logger.info("{} WINDOW REPORT", reportMod);
                    logger.info("MAX-SCORE: {} MED-SCORE: {} MEAN-SCORE: {}",(int)score.getMax(), MathUtils.round(score.getPercentile(50),scale), MathUtils.round(score.getMean(),scale));
                    logger.info("MAX-STEPS: {} MED-STEPS: {} MEAN-STEPS: {}", (int)steps.getMax() ,MathUtils.round(steps.getPercentile(50),scale), MathUtils.round(steps.getMean(),scale));
                    logger.info("TOTAL REPORT");
                    logger.info("MAX-SCORE: {} MAX-STEPS: {}",(int)scoreTotal.getMax(), (int)stepsTotal.getMax());

                }

                games++;

            }

            if(replayBufferFull) {
                birdTrainer.runEpoch();
                if (birdTrainer.getEpoch() > EXPLORE) {
                    logger.info("train thread was done, terminating game thread");
                    break training;
                }
            }

        }

        trainer.notifyListeners(listener -> listener.onTrainingEnd(trainer));

    }

    public static void test(Model model) {
        FlappyBird game = new FlappyBird(NDManager.newBaseManager(), 1, 1, true);
        DefaultTrainingConfig config = setupTrainingConfig();
        try (Trainer trainer = model.newTrainer(config)) {
            RlAgent agent = new QAgent(trainer, REWARD_DISCOUNT);
            while (true) {
                game.runEnvironment(agent, false);
            }
        }
    }


    public static SequentialBlock getBlock() {
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

    public static DefaultTrainingConfig setupTrainingConfig() {
        return new DefaultTrainingConfig(Loss.l2Loss())
                .addTrainingListeners(TrainingListener.Defaults.basic())
                .optOptimizer(Adam.builder().optLearningRateTracker(Tracker.fixed(1e-6f)).build())
                .addEvaluator(new Accuracy())
                .optInitializer(new NormalInitializer(), Parameter.Type.WEIGHT)
                ;
    }
}
