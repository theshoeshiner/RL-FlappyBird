package com.kingyu.rlbird.game.component;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import com.kingyu.rlbird.util.Constant;
import com.kingyu.rlbird.util.GameUtil;


public class Pipe {

    static BufferedImage[] images;

    // NOTE Must do this first TODO rearrange all this static logic to game initialization
    static {
        final int PIPE_IMAGE_COUNT = 3;
        images = new BufferedImage[PIPE_IMAGE_COUNT];
        for (int i = 0; i < PIPE_IMAGE_COUNT; i++) {
            images[i] = GameUtil.loadBufferedImage(Constant.PIPE_IMG_PATH[i]);
        }
    }

    public static final int PIPE_WIDTH = images[0].getWidth();
    public static final int PIPE_HEIGHT = images[0].getHeight();
    public static final int PIPE_HEAD_WIDTH = images[1].getWidth();
    public static final int PIPE_HEAD_HEIGHT = images[1].getHeight();

    private int x, y;
    private final int width;
    private int height;

    boolean visible;

    int type;
    public static final int TYPE_TOP_NORMAL = 0;
    public static final int TYPE_BOTTOM_NORMAL = 1;
    private final int velocity;
    Rectangle pipeCollisionRect;

    public Pipe() {
        this.velocity = Constant.GAME_SPEED;
        this.width = PIPE_WIDTH;
        pipeCollisionRect = new Rectangle();
        pipeCollisionRect.width = PIPE_WIDTH;
    }


    public void setAttribute(int x, int y, int height, int type, boolean visible) {
        this.x = x;
        this.y = y;
        this.height = height;
        this.type = type;
        this.visible = visible;
        setRectangle(this.x + 5, this.y, this.height); // 碰撞矩形位置补偿
    }


    public void setRectangle(int x, int y, int height) {
        pipeCollisionRect.x = x;
        pipeCollisionRect.y = y;
        pipeCollisionRect.height = height;
    }

    public void draw(Graphics g, Bird bird) {
        switch (this.type) {
            case TYPE_TOP_NORMAL:
                drawTopNormal(g);
                break;
            case TYPE_BOTTOM_NORMAL:
                drawBottomNormal(g);
                break;
        }
        if (bird.isDead()) {
            return;
        }
        movement();
//        g.setColor(Color.white);
//        g.drawRect((int) pipeRect.getX(), (int) pipeRect.getY(), (int) pipeRect.getWidth(), (int) pipeRect.getHeight());
    }

    public static final int TOP_PIPE_LENGTHENING = 100;

    private void drawTopNormal(Graphics g) {
        int count = (height - PIPE_HEAD_HEIGHT) / PIPE_HEIGHT + 1;
        for (int i = 0; i < count; i++) {
            g.drawImage(images[0], x, y + i * PIPE_HEIGHT, null);
        }
        g.drawImage(images[1], x - ((PIPE_HEAD_WIDTH - width) >> 1),
                height - TOP_PIPE_LENGTHENING - PIPE_HEAD_HEIGHT, null);
    }

    private void drawBottomNormal(Graphics g) {
        int count = (height - PIPE_HEAD_HEIGHT - Ground.GROUND_HEIGHT) / PIPE_HEIGHT + 1;
        for (int i = 0; i < count; i++) {
            g.drawImage(images[0], x, Constant.FRAME_HEIGHT - PIPE_HEIGHT - Ground.GROUND_HEIGHT - i * PIPE_HEIGHT,
                    null);
        }
        g.drawImage(images[2], x - ((PIPE_HEAD_WIDTH - width) >> 1), Constant.FRAME_HEIGHT - height, null);
    }

    private void movement() {
        x -= velocity;
        pipeCollisionRect.x -= velocity;
        if (x < -1 * PIPE_HEAD_WIDTH) {
            visible = false;
        }
    }


    public boolean isInFrame() {
        return x + width < Constant.FRAME_WIDTH;
    }

    public int getX() {
        return x;
    }

    public Rectangle getPipeCollisionRect() {
        return pipeCollisionRect;
    }

    public boolean isVisible() {
        return visible;
    }

    static class PipePool {
        private static final List<Pipe> pool = new ArrayList<>();

        public static final int FULL_PIPE = (Constant.FRAME_WIDTH
                / (Pipe.PIPE_HEAD_WIDTH + GameElementLayer.HORIZONTAL_INTERVAL) + 2) * 2;
        public static final int MAX_PIPE_COUNT = 30; // 对象池中对象的最大个数

        static {
            for (int i = 0; i < FULL_PIPE; i++) {
                pool.add(new Pipe());
            }
        }


        public static Pipe get() {
            int size = pool.size();
            if (size > 0) {
                return pool.remove(size - 1);
            } else {
                return new Pipe();
            }
        }


        public static void giveBack(Pipe pipe) {
            if (pool.size() < MAX_PIPE_COUNT) {
                pool.add(pipe);
            }
        }
    }

}
