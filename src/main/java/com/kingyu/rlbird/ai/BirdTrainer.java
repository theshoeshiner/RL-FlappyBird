package com.kingyu.rlbird.ai;

import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.training.Trainer;
import com.kingyu.rlbird.game.FlappyBird;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.invoke.MethodHandles;

public class BirdTrainer {

    private static final Logger logger = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    private final RlAgent agent;
    private final Trainer trainer;
    private final FlappyBird game;
    private int epoch = 0;
    boolean started = false;
    long start;

    public BirdTrainer(RlAgent agent, Trainer trainer, FlappyBird game) {
        this.agent = agent;
        this.trainer = trainer;
        this.game = game;
    }

    public void start() {
        started = true;
        start = System.currentTimeMillis();
    }

    public void runEpoch() {


        RlEnv.Step[] batchSteps = game.getBatch();
        agent.trainBatch(batchSteps);
        trainer.step();


        if(epoch % 100 == 0) {
            long dur = System.currentTimeMillis() - start;
            logger.info("EPOCH: {} RATE: {} epochs/s" ,epoch,(epoch* 1000L)/dur);
        }

        trainer.notifyListeners(listener -> listener.onEpoch(trainer));

        epoch++;
    }

    public int getEpoch() {
        return epoch;
    }
}