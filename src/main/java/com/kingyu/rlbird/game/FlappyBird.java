package com.kingyu.rlbird.game;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.LruReplayBuffer;
import ai.djl.modality.rl.ReplayBuffer;
import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import com.kingyu.rlbird.game.component.Bird;
import com.kingyu.rlbird.game.component.GameElementLayer;
import com.kingyu.rlbird.game.component.Ground;
import com.kingyu.rlbird.game.component.ScoreCounter;
import com.kingyu.rlbird.util.Constant;
import com.kingyu.rlbird.util.GameUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.Frame;
import java.awt.Graphics;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;

import static com.kingyu.rlbird.ai.Main.INPUT_FRAMES;
import static com.kingyu.rlbird.util.Constant.FPS;
import static com.kingyu.rlbird.util.Constant.FRAME_HEIGHT;
import static com.kingyu.rlbird.util.Constant.FRAME_WIDTH;
import static com.kingyu.rlbird.util.Constant.FRAME_X;
import static com.kingyu.rlbird.util.Constant.FRAME_Y;
import static com.kingyu.rlbird.util.Constant.GAME_TITLE;


public class FlappyBird extends Frame implements RlEnv {

    private static final Logger logger = LoggerFactory.getLogger(FlappyBird.class);

    private final Ground ground;
    private final Bird bird;
    private final GameElementLayer gameElement;
    private final ScoreCounter scoreCounter;
    private final boolean withGraphics;

    public final NDManager manager;

    public ReplayBuffer getReplayBuffer() {
        return replayBuffer;
    }

    private final ReplayBuffer replayBuffer;
    public BufferedImage currentImg;

    private final float startReward = 0.2f;
    public boolean currentTerminal = false;
    public float currentReward = startReward;

    public int globalStep = 0;
    private int gameStep = 0;

    // These frames have independent managers since they are used across many steps
    public final Queue<NDArray> frameQueue = new ArrayDeque<>(INPUT_FRAMES);

    private FlappyBirdStep currentStep;
    private FlappyBirdStep lastStep;
    private List<FlappyBirdStep> gameSteps = new ArrayList<>();


    public NDManager getManager() {
        return manager;
    }

    public FlappyBird(NDManager manager, int batchSize, int replayBufferSize, boolean withGraphics) {
        this.manager = manager;
        this.replayBuffer =  new LruReplayBuffer(batchSize, replayBufferSize);
        this.withGraphics = withGraphics;
        if (this.withGraphics) {
            initFrame();
            this.setVisible(true);
        }

        currentImg = new BufferedImage(FRAME_WIDTH, FRAME_HEIGHT, BufferedImage.TYPE_4BYTE_ABGR);

        // Prefill observation frames array
        for (int i = 0; i < INPUT_FRAMES; i++) {
            NDArray observation = GameUtil.imgPreprocess(currentImg, NDManager.newBaseManager());
            frameQueue.offer(observation);
        }
        NDManager firstStepManager = NDManager.newBaseManager();
        globalStep = -1;

        currentStep = new FlappyBirdStep(this,firstStepManager, null, null);

        globalStep++;
        ground = new Ground();
        scoreCounter = new ScoreCounter(this);
        bird = new Bird(this);
        gameElement = new GameElementLayer(this);

    }



    @Override
    public ai.djl.modality.rl.env.RlEnv.Step step(NDList action, boolean training) {

        NDManager stepManager = manager.newSubManager();
        lastStep = currentStep;

        // action is either flap or nothing
        if (action.singletonOrThrow().getInt(1) == 1) {
            bird.birdFlap();
        }
        stepFrame();
        if (this.withGraphics) {
            repaint();
            try {
                Thread.sleep(FPS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }


        FlappyBirdStep step = new FlappyBirdStep(this,stepManager, action, lastStep);

/*     if (training) {
            // This will close old steps automatically, so we sync on replay buffer to ensure that no steps are returned while we are deciding what to close
            synchronized (replayBuffer) {
                replayBuffer.addStep(step);
            }
        }*/

        currentStep = step;

        globalStep++;
        gameStep++;


        gameSteps.add(step);
        return step;
    }

    @Override
    public float runEnvironment(RlAgent agent, boolean training) {
        float reward =  RlEnv.super.runEnvironment(agent, training);
        if(training) {
            // add steps all at once to the buffer
            synchronized (replayBuffer) {
                gameSteps.stream().forEach(replayBuffer::addStep);
                gameSteps.clear();
            }
        }
        return reward;
    }

    @Override
    public NDList getObservation() {
        return currentStep.getPostObservation();
    }

    @Override
    public ActionSpace getActionSpace() {
        return currentStep.getPostActionSpace();
    }


    @Override
    public ai.djl.modality.rl.env.RlEnv.Step[] getBatch() {
        return replayBuffer.getBatch();
    }


    @Override
    public void close() {
        manager.close();
    }


    @Override
    public void reset() {
        gameElement.reset();
        bird.reset();
        currentReward = startReward;
        currentTerminal = false;
        gameStep = 0;
    }



    public void addFrame() {
        frameQueue.remove().getManager().close();
        NDArray observation = GameUtil.imgPreprocess(currentImg, NDManager.newBaseManager());
        frameQueue.offer(observation);
    }


    public void stepFrame() {
        Graphics bufG = currentImg.getGraphics();
        bufG.setColor(Constant.BG_COLOR);
        bufG.fillRect(0, 0, Constant.FRAME_WIDTH, Constant.FRAME_HEIGHT);
        ground.draw(bufG, bird);
        bird.draw(bufG);
        gameElement.draw(bufG, bird);
        addFrame();

    }

    private void initFrame() {
        setSize(FRAME_WIDTH, FRAME_HEIGHT);
        setTitle(GAME_TITLE);
        setLocation(FRAME_X, FRAME_Y);
        setResizable(false);
        setVisible(true);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        });
    }


    @Override
    public void update(Graphics g) {
        g.drawImage(currentImg, 0, 0, null);
    }

    public void setCurrentTerminal(boolean currentTerminal) {
        this.currentTerminal = currentTerminal;
    }

    public void setCurrentReward(float currentReward) {
        this.currentReward = currentReward;
    }

    public long getScore() {
        return scoreCounter.getCurrentScore();
    }

    public Bird getBird() {
        return bird;
    }

    public GameElementLayer getGameElement() {
        return gameElement;
    }

    public ScoreCounter getScoreCounter(){
        return scoreCounter;
    }

    public int getGameStep() {
        return gameStep;
    }
}
