package com.kingyu.rlbird.ai;

import ai.djl.Model;
import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.ndarray.NDManager;
import ai.djl.training.Trainer;
import com.kingyu.rlbird.game.FlappyBird;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math.util.MathUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.invoke.MethodHandles;

public class GameThread extends Thread {

    public static final int START_TRAINING_AFTER = 1000; // fill the buffer with this many steps before beginning training

    private static final Logger logger = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    private final RlAgent agent;
    //private final Trainer trainer;
    //private final Model model;
    private final FlappyBird game;
    private final NDManager gameManager;
    private final TrainerThread trainerThread;


    boolean started = false;
    long start;

    public GameThread(RlAgent agent, Trainer trainer, Model model, FlappyBird game, TrainerThread trainerThread) {
        this.agent = agent;
        //this.trainer = trainer;
        //this.model = model;
        this.game = game;
        this.gameManager = game.getManager();
        this.trainerThread = trainerThread;
    }

    public void run() {

        int reportEveryGames = 30;
        int scale = 3;
        long reportStart = System.currentTimeMillis();
        long reportSteps = 0;
        long reportGames = 0;

        DescriptiveStatistics score = new DescriptiveStatistics(reportEveryGames);
        DescriptiveStatistics steps = new DescriptiveStatistics(reportEveryGames);
        DescriptiveStatistics scoreTotal = new DescriptiveStatistics();
        DescriptiveStatistics stepsTotal = new DescriptiveStatistics();

        long totalSteps = 0;
        //BirdTrainer birdTrainer = new BirdTrainer(model,agent,trainer,game);
        boolean trainingStarted = false;
        int games = 0; // game == episode

        while(true) {

           // int batchSteps = 0;

            // Make sure we generate at least enough steps to cover the last processed batch
            //while(batchSteps < BATCH_SIZE) {
            try {
                Thread.sleep(0);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }

            //synchronized (game.getReplayBuffer()) { // need to sync on this since we will be closing steps here
            // Dont need to sync on buffer since that will be done by the game instance
            float result = game.runEnvironment(agent, true); // runs game until a terminal step, training is assumed to be true
            //
            int gameSteps = game.getGameStep();
            long gameScore = game.getScore();
            //batchSteps += gameSteps;

            score.addValue(gameScore);
            steps.addValue(gameSteps);
            scoreTotal.addValue(gameScore);
            stepsTotal.addValue(gameSteps);

            totalSteps += gameSteps;
            reportSteps += gameSteps;
            reportGames++;
            games++;
            //reportGames += game.gameStep;

            if (!trainingStarted && totalSteps > START_TRAINING_AFTER) {
                trainingStarted = true;
                trainerThread.start();

            }

            if (games % reportEveryGames == 0) {
                long dur = System.currentTimeMillis() - reportStart;

                logger.info("GAME: {} TOTAL-STEPS: {} ", games, totalSteps);
                logger.info("RATE: {} steps/s {} games/s", (reportSteps * 1000L) / dur,(reportGames * 1000L) / dur);


                logger.info("ARRAYS {}", gameManager.getManagedArrays().size());
                logger.info("WINMX REPORT --> MAX-SCORE: {} MAX-STEPS: {}", (int)score.getMax(),(int)steps.getMax());
                logger.info("WINMD REPORT --> MED-SCORE: {} MED-STEPS: {}",  MathUtils.round(score.getPercentile(50),scale), MathUtils.round(steps.getPercentile(50),scale));
                logger.info("WINMN REPORT --> MEN-SCORE: {} MEN-STEPS: {}",  MathUtils.round(score.getMean(),scale), MathUtils.round(steps.getMean(),scale));
                logger.info("TOTAL REPORT --> MAX-SCORE: {} MAX-STEPS: {}",(int)scoreTotal.getMax(), (int)stepsTotal.getMax());

                reportStart = System.currentTimeMillis();
                reportGames = 0;
                reportSteps = 0;

            }


            if(trainingStarted && !trainerThread.isAlive()) {
                logger.info("train thread was done, terminating game thread");
                break;
            }

        }

        this.interrupt();

    }
}
