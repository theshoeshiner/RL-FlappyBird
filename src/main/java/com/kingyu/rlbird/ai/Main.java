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
import com.kingyu.rlbird.util.Arguments;
import com.kingyu.rlbird.util.Constant;
import org.apache.commons.cli.ParseException;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Paths;

public final class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    //public static final int EXPLORE = 3_000_000; // frames over which to anneal epsilon og was 3_000_000
    //public static final int STEPS_PER_EPISODE = 3_000_000; // frames over which to anneal epsilon og was 3_000_000
    public static final int SAVE_EVERY_STEPS = 100; // save model every 1,000 epochs
    public static final int REPLAY_BUFFER_SIZE = 5_000; // number of previous transitions to remember TODO DCW Org version was 50k
    public static final float REWARD_DISCOUNT = 0.9f; // decay rate of past observations
    //public static final String PARAMS_PREFIX = "dqn-trained";

    public static final float INITIAL_EPSILON = 0.01f; //0.01
    public static final float FINAL_EPSILON = 0.0001f; // 0.0001



    public static int INPUT_FRAMES = 4;
    public static int SCREEN_SIZE = 80;
    public static int BATCH_SIZE = 32;
    public static String modelNamePrefix;

    private Main() {}

    public static void main(String[] args) throws ParseException, IOException, MalformedModelException {

        try {

            Arguments arguments = Arguments.parseArgs(args);

            //public FlappyBirdTrainer(int inputFrames, int batchSize, int bufferSize, float startEpsilon, float endEpsilon, float rewardDiscount) {
            FlappyBirdTrainer trainer = new FlappyBirdTrainer(INPUT_FRAMES, BATCH_SIZE, REPLAY_BUFFER_SIZE, INITIAL_EPSILON, FINAL_EPSILON, REWARD_DISCOUNT, arguments.usePreTrained(), arguments.withGraphics(), arguments.isTesting());

            trainer.run();
          /*  modelNamePrefix = "dqn-replay_" +REPLAY_BUFFER_SIZE+"-batch_"+BATCH_SIZE;

            Model model = createOrLoadModel(arguments);
            if (arguments.isTesting()) {
                test(model);
            } else {
                train(arguments, model);
            }*/

        }
        catch(Exception e){
            e.printStackTrace();
        }
    }

  /*  public static Model createOrLoadModel(Arguments arguments) throws MalformedModelException {
        Model model = Model.newInstance("QNetwork");

        model.setBlock(getBlock());
        if (arguments.usePreTrained()) {
            logger.info("loading model: {}",modelNamePrefix);
            try {
                model.load(Paths.get(Constant.MODEL_PATH), modelNamePrefix);
                logger.info("loaded model at epoch {}",model.getProperty("Epoch"));
            } catch (IOException ignored) {
            }


        }
        return model;
    }*/

    /*public static void train(Arguments arguments, Model model) {

        logger.info("REWARD_DISCOUNT: {}",REWARD_DISCOUNT);
        logger.info("INITIAL_EPSILON: {}",INITIAL_EPSILON);
        logger.info("FINAL_EPSILON: {}",FINAL_EPSILON);
        logger.info("REPLAY_BUFFER_SIZE: {}",REPLAY_BUFFER_SIZE);
        logger.info("BATCH_SIZE: {}",BATCH_SIZE);



        boolean withGraphics = arguments.withGraphics();


        NDManager gameManager = NDManager.newBaseManager();

        FlappyBird game = new FlappyBird(gameManager, BATCH_SIZE, REPLAY_BUFFER_SIZE, withGraphics);


        DefaultTrainingConfig config = setupTrainingConfig();
        Trainer trainer = model.newTrainer(config);

        trainer.initialize(new Shape(BATCH_SIZE, INPUT_FRAMES, 80, 80));

       *//* trainer.initialize(
                new Shape(batchSize, INPUT_FRAMES, SCREEN_SIZE, SCREEN_SIZE), // state in,
                //new Shape(batchSize), // action space?
                new Shape(batchSize, 2) // action - flap or dont

        );*//*


        QAgent a;


        RlAgent agent = new QAgent(trainer, REWARD_DISCOUNT);
        Tracker exploreRate =
                LinearTracker.builder()
                        .setBaseValue(INITIAL_EPSILON)
                        .optSlope(-(INITIAL_EPSILON - FINAL_EPSILON) / TrainerThread.TRAIN_STEPS)
                        .optMinValue(FINAL_EPSILON)
                        .build();
        agent = new EpsilonGreedy(agent, exploreRate);

        int reportMod = 100;
        int scale = 3;
        long reportStart = System.currentTimeMillis();
        long reportSteps = 0;

        DescriptiveStatistics score = new DescriptiveStatistics(reportMod);
        DescriptiveStatistics steps = new DescriptiveStatistics(reportMod);
        DescriptiveStatistics scoreTotal = new DescriptiveStatistics();
        DescriptiveStatistics stepsTotal = new DescriptiveStatistics();

        long totalSteps = 0;
        TrainerThread trainerThread = new TrainerThread(model,agent,trainer,game);
        boolean replayBufferFull = false;
        int games = 0; // game == episode

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

        *//*training: while(true) {

            int batchSteps = 0;

            // Make sure we generate at least enough steps to cover the last processed batch
            while(batchSteps < BATCH_SIZE) {

                float result = game.runEnvironment(agent, true); // runs game until a terminal step, training is assumed to be true
                int gameSteps = game.getGameStep();
                batchSteps += gameSteps;

                score.addValue(game.getScore());
                steps.addValue(game.gameStep);
                scoreTotal.addValue(game.getScore());
                stepsTotal.addValue(game.gameStep);

                totalSteps += game.gameStep;
                reportSteps += game.gameStep;
                if (!replayBufferFull && totalSteps > REPLAY_BUFFER_SIZE) {
                    replayBufferFull = true;
                    birdTrainer.start();

                }

                if (games % reportMod == 0) {
                    long dur = System.currentTimeMillis() - reportStart;

                    logger.info("GAME: {} TOTAL-STEPS: {} ", games, totalSteps);
                    logger.info("RATE: {} steps/s {} games/s", (reportSteps * 1000L) / dur,(reportMod * 1000L) / dur);

                    logger.info("ARRAYS {}", gameManager.getManagedArrays().size());
                    logger.info("WINMX REPORT --> MAX-SCORE: {} MAX-STEPS: {}", (int)score.getMax(),(int)steps.getMax());
                    logger.info("WINMD REPORT --> MED-SCORE: {} MED-STEPS: {}",  MathUtils.round(score.getPercentile(50),scale), MathUtils.round(steps.getPercentile(50),scale));
                    logger.info("WINMN REPORT --> MEN-SCORE: {} MEN-STEPS: {}",  MathUtils.round(score.getMean(),scale), MathUtils.round(steps.getMean(),scale));
                    logger.info("TOTAL REPORT --> MAX-SCORE: {} MAX-STEPS: {}",(int)scoreTotal.getMax(), (int)stepsTotal.getMax());

                    reportStart = System.currentTimeMillis();
                    reportSteps = 0;
                }

                games++;

            }

            if(replayBufferFull) {
                birdTrainer.runBatch();
                if (birdTrainer.getBatch() > EXPLORE) {
                    logger.info("train thread was done, terminating game thread");
                    break training;
                }



            }

        }

        trainer.notifyListeners(listener -> listener.onTrainingEnd(trainer));
*//*
    }*/

/*
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
*/


    /*public static SequentialBlock getBlock() {
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
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                .addTrainingListeners(TrainingListener.Defaults.basic())
                .optOptimizer(Adam.builder().optLearningRateTracker(Tracker.fixed(1e-6f)).build())
                .addEvaluator(new Accuracy())
                .optInitializer(new NormalInitializer(), Parameter.Type.WEIGHT)
                ;
        SaveModelTrainingListener saveModelTrainingListener = new SaveModelTrainingListener(Paths.get(Constant.MODEL_PATH).toString(),modelNamePrefix,SAVE_EVERY_STEPS);
        saveModelTrainingListener.setSaveModelCallback(trainer -> {
            logger.info("saving model with epoch: {}",trainer.getModel().getProperty("Epoch"));
        });
        config.addTrainingListeners(saveModelTrainingListener);
        return config;
    }*/
}
