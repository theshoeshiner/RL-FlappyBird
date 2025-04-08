package com.kingyu.rlbird.ai;

import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.invoke.MethodHandles;
import java.util.HashMap;
import java.util.Map;

public class NDManagerTracker {

    private static final Logger logger = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

    private final Map<String,Integer> preMap = new HashMap<>();
    private final Map<String,Integer> postMap = new HashMap<>();
    private final NDManager manager;

    public NDManagerTracker(NDManager manager) {
        this.manager = manager;
    }

    public void pre(String name){
        int current = manager.getManagedArrays().size();
        logger.info("{} started with {} global arrays",name, current);
        //((BaseNDManager)manager).debugDump(0);
        preMap.put(name,current);
    }

    public void post(String name){
        int current = manager.getManagedArrays().size();
        postMap.put(name,current);
        Integer pre = preMap.get(name);
        //((BaseNDManager)manager).debugDump(0);
        logger.info("{} created {} arrays",name, current-pre);
    }

}
