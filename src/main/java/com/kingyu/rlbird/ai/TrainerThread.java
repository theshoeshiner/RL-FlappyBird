package com.kingyu.rlbird.ai;

import ai.djl.Model;
import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.training.Trainer;
import com.kingyu.rlbird.game.FlappyBird;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.invoke.MethodHandles;

public class TrainerThread extends Thread {

    private static final Logger logger = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    public static final int TRAIN_STEPS = 3_000_000; // total number of training batches to run
    public static final int EPOCH_BATCHES = 5000; // batches per epoch
    // Epoch doesnt really have a valid meaning for RL, but we can use it to control how often we save the model
    public static final int EPOCHS = TRAIN_STEPS/EPOCH_BATCHES;

    //public static final int STEPS_PER_EPISODE = 3_000_000; // frames over which to anneal epsilon og was 3_000_000
    //public static final int SAVE_EVERY_STEPS = 100; // save model every 1,000 epochs

    private static final int REPORT_EVERY_BATCHES = 500;

    private static final int REPORT_EVERY_EPOCHS = 1;

    private final RlAgent agent;
    private final Trainer trainer;
    private final Model model;
    private final FlappyBird game;
    private int batch = 0;
    private int epoch = 0;
    private int step = 0;
    boolean started = false;
    long start;


    public TrainerThread(Model model, RlAgent agent, Trainer trainer, FlappyBird game) {
        this.agent = agent;
        this.trainer = trainer;
        this.game = game;
        this.model = model;
        //this.batch = model.intProperty("Epoch",0);
        epoch = model.intProperty("Epoch",0);
    }


    @Override
    public void run() {
        started = true;
        start = System.currentTimeMillis();


        logger.info("Starting training");

        trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));


        // want to run the train thread through a specific number of episodes
        while(epoch<EPOCHS){
            for(int b=0;b<EPOCH_BATCHES;b++){

                synchronized (game.getReplayBuffer()) { // need to sync to prevent steps being closed
                    RlEnv.Step[] batchSteps = game.getBatch();
                    //logger.info("bacthsteps: {}", batchSteps.length);
                    agent.trainBatch(batchSteps);
                    //logger.info("training done");
                    trainer.step();
                    batch++;
                    step += batchSteps.length;

                    if(batch % REPORT_EVERY_BATCHES == 0) {
                        long dur = System.currentTimeMillis() - start;
                        logger.info("BATCH: {} RATE: {} batches/s" , batch,(batch * 1000L)/dur);

                    }

                }
            }


            if(epoch % REPORT_EVERY_EPOCHS == 0) {
                long dur = System.currentTimeMillis() - start;
                logger.info("EPOCH: {} RATE: {} epochs/s" , epoch,(epoch * 1000L)/dur);

            }

            epoch++;


            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
        }


        trainer.notifyListeners(listener -> listener.onTrainingEnd(trainer));

    }

/*
    public void runBatch() {

        for(int e=0;e<EPOCHS;e++){

            for(int b=0;b<EPOCH_BATCHES;b++){

                RlEnv.Step[] batchSteps = game.getBatch();
                agent.trainBatch(batchSteps);
                trainer.step();
                batch++;
                step+=batchSteps.length;
                if(batch % 100 == 0) {
                    long dur = System.currentTimeMillis() - start;
                    logger.info("BATCH: {} RATE: {} batches/s" , batch,(batch * 1000L)/dur);
                }

            }
            epoch++;
            trainer.notifyListeners(listener -> listener.onEpoch(trainer));
        }


    }*/

    public int getBatch() {
        return batch;
    }
}