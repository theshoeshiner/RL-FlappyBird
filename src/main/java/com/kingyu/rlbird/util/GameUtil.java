package com.kingyu.rlbird.util;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import com.kingyu.rlbird.ai.TrainBird;

import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.IOException;

import javax.imageio.ImageIO;


public class GameUtil {

    private GameUtil() {
    }

    public static BufferedImage loadBufferedImage(String imgPath) {
        try {
            return ImageIO.read(new FileInputStream(imgPath));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }


    public static int getRandomNumber(int min, int max) {
        return (int) (Math.random() * (max - min) + min);
    }


    public static NDArray imgPreprocess(BufferedImage observation, NDManager manager) {
        return NDImageUtils.toTensor(
                NDImageUtils.resize(
                        ImageFactory.getInstance().fromImage(observation).toNDArray(manager, Image.Flag.GRAYSCALE)
                        , TrainBird.SCREEN_SIZE,TrainBird.SCREEN_SIZE));
    }
}