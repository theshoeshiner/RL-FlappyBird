package com.kingyu.rlbird.rl.agent;

import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListener.BatchData;
import ai.djl.translate.Batchifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;

/**
 * An {@link RlAgent} that implements Q or Deep-Q Learning.
 *
 * <p>Deep-Q Learning estimates the total reward that will be given until the environment ends in a
 * particular state after taking a particular action. Then, it is trained by ensuring that the
 * prediction before taking the action match what would be predicted after taking the action. More
 * information can be found in the <a
 * href="https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf">paper</a>.
 *
 * <p>It is one of the earliest successful techniques for reinforcement learning with Deep learning.
 * It is also a good introduction to the field. However, many better techniques are commonly used
 * now.
 */
public class QAgent extends ai.djl.modality.rl.agent.QAgent {

    private static final Logger logger = LoggerFactory.getLogger(QAgent.class);
    private final Trainer trainer;
    private final float rewardDiscount;
    // This batchifier makes the super.chooseAgent method functionally equivalent to the pre-upgrade code
    private final static Batchifier batchifier = new Batchifier() {
        @Override
        public NDList batchify(NDList[] inputs) {
            return new NDList(inputs[0].get(0));
        }

        @Override
        public NDList[] unbatchify(NDList inputs) {
            return new NDList[0];
        }
    };

    /**
     * Constructs a {@link ai.djl.modality.rl.agent.QAgent} with a custom {@link Batchifier}.
     *
     * @param trainer        the trainer for the model to learn
     * @param rewardDiscount the reward discount to apply to rewards from future states
     */
    public QAgent(Trainer trainer, float rewardDiscount) {
        super(trainer, rewardDiscount,batchifier );
        this.trainer = trainer;
        this.rewardDiscount = rewardDiscount;

    }

    @Override
    public void trainBatch(ai.djl.modality.rl.env.RlEnv.Step[] batchSteps) {

       /* BatchData batchData =
                new BatchData(null, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());
        for (RlEnv.Step step : batchSteps) {

            NDList[] preInput = buildInputs(step.getPreObservation(), Collections.singletonList(step.getAction()));


            NDList[] postInputs = buildInputs(step.getPostObservation(), step.getPostActionSpace());

            NDList[] allInputs =
                    Stream.concat(Arrays.stream(preInput), Arrays.stream(postInputs))
                            .toArray(NDList[]::new);

            try (GradientCollector collector = trainer.newGradientCollector()) {
                NDArray results =
                        trainer.forward(batchifier.batchify(allInputs))
                                .singletonOrThrow()
                                .squeeze(-1);
                NDList preQ = new NDList(results.get(0));
                NDList postQ;
                if (step.isDone()) {
                    postQ = new NDList(step.getReward());
                } else {
                    NDArray bestAction = results.get("1:").max();
                    postQ = new NDList(bestAction.mul(rewardDiscount).add(step.getReward()));
                }
                NDArray lossValue = trainer.getLoss().evaluate(postQ, preQ);
                collector.backward(lossValue);
                batchData.getLabels().put(postQ.get(0).getDevice(), postQ);
                batchData.getPredictions().put(preQ.get(0).getDevice(), preQ);
            }
        }

        trainer.notifyListeners(listener -> listener.onTrainingBatch(trainer, batchData));*/


        // Working

        BatchData batchData = new BatchData(null, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());

        NDList preObservationBatch = new NDList();
        Arrays.stream(batchSteps).forEach(step -> preObservationBatch.addAll(step.getPreObservation()));
        NDList preInput = new NDList(NDArrays.concat(preObservationBatch, 0));

        NDList postObservationBatch = new NDList();
        Arrays.stream(batchSteps).forEach(step -> postObservationBatch.addAll(step.getPostObservation()));
        NDList postInput = new NDList(NDArrays.concat(postObservationBatch, 0));

        NDList actionBatch = new NDList();
        Arrays.stream(batchSteps).forEach(step -> actionBatch.addAll(step.getAction()));
        NDList actionInput = new NDList(NDArrays.stack(actionBatch, 0));

        NDList rewardBatch = new NDList();
        Arrays.stream(batchSteps).forEach(step -> rewardBatch.addAll(new NDList(step.getReward())));
        NDList rewardInput = new NDList(NDArrays.stack(rewardBatch, 0));

        try (GradientCollector collector = trainer.newGradientCollector()) {

            NDList preResults = trainer.forward(preInput);
            NDList postResults = trainer.forward(postInput);

            NDList preQ = new NDList(preResults.singletonOrThrow()
                    .mul(actionInput.singletonOrThrow())
                    .sum(new int[]{1}));

            NDArray[] postQ = new NDArray[batchSteps.length];
            for (int i = 0; i < batchSteps.length; i++) {
                if (batchSteps[i].isDone()) {
                    postQ[i] = batchSteps[i].getReward();
                } else {
                    postQ[i] = postResults.singletonOrThrow().get(i)
                            .max()
                            .mul(rewardDiscount)
                            .add(rewardInput.singletonOrThrow().get(i));
                }
            }
            NDList targetQBatch = new NDList();
            Arrays.stream(postQ).forEach(value -> targetQBatch.addAll(new NDList(value)));
            NDList targetQ = new NDList(NDArrays.stack(targetQBatch, 0));

            NDArray lossValue = trainer.getLoss().evaluate(targetQ, preQ);
            collector.backward(lossValue);
            batchData.getLabels().put(targetQ.singletonOrThrow().getDevice(), targetQ);
            batchData.getPredictions().put(preQ.singletonOrThrow().getDevice(), preQ);
        }

        // broken for old code
        //trainer.notifyListeners(listener -> listener.onTrainingBatch(trainer, batchData));

    }

}