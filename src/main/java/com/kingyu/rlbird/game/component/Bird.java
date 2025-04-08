package com.kingyu.rlbird.game.component;

import java.awt.*;
import java.awt.image.BufferedImage;

import com.kingyu.rlbird.game.FlappyBird;
import com.kingyu.rlbird.util.Constant;
import com.kingyu.rlbird.util.GameUtil;


public class Bird {
    private final int x;
    private int y;

    private int birdState;
    public static final int BIRD_READY = 0;
    public static final int BIRD_FALL = 1;
    public static final int BIRD_DEAD = 2;

    private final Rectangle birdCollisionRect;
    public static final int RECT_DESCALE = 2;

    static BufferedImage birdImages;

    public static final int BIRD_WIDTH;
    public static final int BIRD_HEIGHT;

    static {
        birdImages = GameUtil.loadBufferedImage(Constant.BIRDS_IMG_PATH);
        assert birdImages != null;
        BIRD_WIDTH = birdImages.getWidth();
        BIRD_HEIGHT = birdImages.getHeight();
    }

    final FlappyBird game;

    public Bird(FlappyBird game) {
        this.game = game;

        x = Constant.FRAME_WIDTH >> 2;
        y = Constant.FRAME_HEIGHT >> 1;

        int rectX = x - (BIRD_WIDTH >> 1);
        int rectY = y - (BIRD_HEIGHT >> 1) + RECT_DESCALE * 2;
        birdCollisionRect = new Rectangle(rectX + RECT_DESCALE, rectY + RECT_DESCALE * 2, BIRD_WIDTH - RECT_DESCALE * 3,
                BIRD_HEIGHT - RECT_DESCALE * 4);
    }

    public void draw(Graphics g) {
        movement();
        drawBirdImg(g);
//        g.setColor(Color.white);
//        g.drawRect((int) birdCollisionRect.getX(), (int)birdCollisionRect.getY(), (int) birdCollisionRect.getWidth(), (int) birdCollisionRect.getHeight());
    }

    public void drawBirdImg(Graphics g) {
        g.drawImage(birdImages, x - (BIRD_WIDTH >> 1), y - (BIRD_HEIGHT >> 1), null);
    }

    public static final int ACC_FLAP = 15; // players speed on flapping
    public static final double ACC_Y = 4; // players downward acceleration
    public static final int MAX_VEL_Y = -25; // max vel along Y, max descend speed
    public static final int BOTTOM_BOUNDARY = Constant.FRAME_HEIGHT - Ground.GROUND_HEIGHT - (BIRD_HEIGHT >> 1);

    private int velocity = 0; // bird's velocity along Y, default same as playerFlapped

    private void movement() {
        if (velocity > MAX_VEL_Y)
            velocity -= ACC_Y;
        y = Math.min((y - velocity), BOTTOM_BOUNDARY);
        birdCollisionRect.y = birdCollisionRect.y - velocity;
        if (birdCollisionRect.y < GameElementLayer.MIN_HEIGHT ||
                birdCollisionRect.y > GameElementLayer.MAX_HEIGHT + GameElementLayer.VERTICAL_INTERVAL) {
            game.setCurrentReward(0.1f);
        }
        if (birdCollisionRect.y < Constant.WINDOW_BAR_HEIGHT) {
            die();
        }
        if (birdCollisionRect.y >= BOTTOM_BOUNDARY - 10) {
            die();
        }
    }

    public void birdFlap() {
        if (isDead())
            return;
        velocity = ACC_FLAP;
    }

    public void die() {
        game.setCurrentReward(-1);
        game.setCurrentTerminal(true);
        birdState = BIRD_DEAD;
    }

    public boolean isDead() {
        return birdState == BIRD_FALL || birdState == BIRD_DEAD;
    }

    public void reset() {
        birdState = BIRD_READY;
        y = Constant.FRAME_HEIGHT >> 1;
        velocity = 0;
        int ImgHeight = birdImages.getHeight();
        birdCollisionRect.y = y + RECT_DESCALE * 4 - ImgHeight / 2;
        game.getScoreCounter().reset();
    }

    public int getBirdX() {
        return x;
    }

    public Rectangle getBirdCollisionRect() {
        return birdCollisionRect;
    }
}
