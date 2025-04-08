package com.kingyu.rlbird.game.component;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

import com.kingyu.rlbird.game.FlappyBird;
import com.kingyu.rlbird.util.Constant;
import com.kingyu.rlbird.util.GameUtil;
import com.kingyu.rlbird.game.component.Pipe.PipePool;


public class GameElementLayer {

    private final List<Pipe> pipes;
    private final FlappyBird game;

    public GameElementLayer(FlappyBird game) {
        this.game = game;
        pipes = new ArrayList<>();
    }

    public void draw(Graphics g, Bird bird) {
        for (int i = 0; i < pipes.size(); i++) {
            Pipe pipe = pipes.get(i);
            if (pipe.isVisible()) {
                pipe.draw(g, bird);
            } else {
                Pipe remove = pipes.remove(i);
                PipePool.giveBack(remove);
                i--;
            }
        }
        bird.drawBirdImg(g);
        isCollideBird(bird);
        generatePipe(bird);
    }

    public static final int VERTICAL_INTERVAL = Constant.FRAME_HEIGHT >> 2;
    public static final int HORIZONTAL_INTERVAL = Constant.FRAME_HEIGHT >> 2;
    public static final int MIN_HEIGHT = Constant.FRAME_HEIGHT / 5;
    public static final int MAX_HEIGHT = Constant.FRAME_HEIGHT / 3;


    private void generatePipe(Bird bird) {
        if (bird.isDead()) {
            return;
        }
        if (pipes.size() == 0) {

            int topHeight = GameUtil.getRandomNumber(MIN_HEIGHT, MAX_HEIGHT + 1);

            Pipe top = PipePool.get();
            top.setAttribute(Constant.FRAME_WIDTH, -Pipe.TOP_PIPE_LENGTHENING,
                    topHeight + Pipe.TOP_PIPE_LENGTHENING, Pipe.TYPE_TOP_NORMAL, true);

            Pipe bottom = PipePool.get();
            bottom.setAttribute(Constant.FRAME_WIDTH, topHeight + VERTICAL_INTERVAL,
                    Constant.FRAME_HEIGHT - topHeight - VERTICAL_INTERVAL, Pipe.TYPE_BOTTOM_NORMAL, true);

            pipes.add(top);
            pipes.add(bottom);
        } else {

            Pipe lastPipe = pipes.get(pipes.size() - 1);
            int currentDistance = lastPipe.getX() - bird.getBirdX() + Bird.BIRD_WIDTH / 2;
            final int SCORE_DISTANCE = Pipe.PIPE_WIDTH * 2 + HORIZONTAL_INTERVAL;
            if (pipes.size() >= PipePool.FULL_PIPE
                    && currentDistance <= SCORE_DISTANCE + Pipe.PIPE_WIDTH * 3 / 2
                    && currentDistance > SCORE_DISTANCE + Pipe.PIPE_WIDTH * 3 / 2 - Constant.GAME_SPEED) {
                game.setCurrentReward(0.8f);
            }
            if (pipes.size() >= PipePool.FULL_PIPE
                    && currentDistance <= SCORE_DISTANCE
                    && currentDistance > SCORE_DISTANCE - Constant.GAME_SPEED) {
                game.getScoreCounter().score();
            }
            if (lastPipe.isInFrame()) {
                addNormalPipe(lastPipe);
            }
        }
    }


    private void addNormalPipe(Pipe lastPipe) {
        int topHeight = GameUtil.getRandomNumber(MIN_HEIGHT, MAX_HEIGHT + 1);
        int x = lastPipe.getX() + HORIZONTAL_INTERVAL;

        Pipe top = PipePool.get();
        top.setAttribute(x, -Pipe.TOP_PIPE_LENGTHENING, topHeight + Pipe.TOP_PIPE_LENGTHENING,
                Pipe.TYPE_TOP_NORMAL, true);

        Pipe bottom = PipePool.get();
        bottom.setAttribute(x, topHeight + VERTICAL_INTERVAL, Constant.FRAME_HEIGHT - topHeight - VERTICAL_INTERVAL,
                Pipe.TYPE_BOTTOM_NORMAL, true);

        pipes.add(top);
        pipes.add(bottom);
    }


    public void isCollideBird(Bird bird) {
        if (bird.isDead()) {
            return;
        }
        for (Pipe pipe : pipes) {
            if (pipe.getPipeCollisionRect().intersects(bird.getBirdCollisionRect())) {
                bird.die();
                return;
            }
        }
    }

    public void reset() {
        for (Pipe pipe : pipes) {
            PipePool.giveBack(pipe);
        }
        pipes.clear();
    }
}
