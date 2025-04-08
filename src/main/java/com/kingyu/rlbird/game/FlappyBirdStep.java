package com.kingyu.rlbird.game;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.stream.Collectors;

import static com.kingyu.rlbird.util.Constant.DO_NOTHING;
import static com.kingyu.rlbird.util.Constant.FLAP;

public final class FlappyBirdStep implements ai.djl.modality.rl.env.RlEnv.Step {

    private static final Logger logger = LoggerFactory.getLogger(FlappyBirdStep.class);

    private final NDList observation;
    private final boolean terminal;
    private final NDManager manager;
    private final int globalStep;
    private final NDArray reward;
    private final ActionSpace actionSpace;
    private final NDList preObservation;
    private final NDList action;


    FlappyBirdStep(FlappyBird game, NDManager postStateManager, NDList action, FlappyBirdStep lastStep) {
        this.manager = postStateManager;

        NDArray framesStacked = NDArrays.stack(game.frameQueue.stream().map(NDArray::duplicate).peek(a -> a.attach(manager)).collect(Collectors.toCollection(NDList::new)), 1);
        observation = new NDList(framesStacked);
        terminal = game.currentTerminal;
        reward = manager.create(game.currentReward);
        globalStep = game.globalStep;

        actionSpace = new ActionSpace();
        actionSpace.add(new NDList(manager.create(DO_NOTHING)));
        actionSpace.add(new NDList(manager.create(FLAP)));

        this.action = action == null ? null : action.stream().map(NDArray::duplicate).peek(a -> a.attach(manager)).collect(Collectors.toCollection(NDList::new));
        this.preObservation = lastStep == null ? null : lastStep.observation.stream().map(NDArray::duplicate).peek(a -> a.attach(manager)).collect(Collectors.toCollection(NDList::new));


    }


    @Override
    public NDList getPreObservation() {
        return preObservation;
    }


    @Override
    public NDList getPostObservation() {
        return observation;
    }

    @Override
    public ActionSpace getPostActionSpace() {
        return actionSpace;
    }


    @Override
    public NDList getAction() {
        return action;
    }


    @Override
    public NDArray getReward() {
        return reward;
    }


    @Override
    public boolean isDone() {
        return terminal;
    }

    public String toString() {
        return "step" + globalStep;
    }

    @Override
    public void close() {
        this.manager.close(); // Closes the manager that contains the current observation
    }
}
