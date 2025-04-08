package com.kingyu.rlbird.game.component;

import com.kingyu.rlbird.game.FlappyBird;


public class ScoreCounter {

    private long score = 0;
    private final FlappyBird game;

    public ScoreCounter(FlappyBird game) {
        this.game = game;
    }

    public void score() {
        if (!game.getBird().isDead()) {
            game.setCurrentReward(1f);
            score += 1;
        }
    }

    public long getCurrentScore() {
        return score;
    }

    public void reset() {
        score = 0;
    }

}
