package pp.facerecognizer.search;

import androidx.core.util.Pair;
import java.util.ArrayList;
import pp.facerecognizer.recognition.MxNetUtils;


public class Search {

    private ArrayList<float[]> storage_list = new ArrayList<>();

    public void storage(int label, ArrayList<float[]> emb_list) {
        if (label + 1 > storage_list.size()) storage_list.add(emb_list.get(0));
        else storage_list.set(label, emb_list.get(0));
    }

    public Pair<Integer, Float> predict(float[] source_emb) {
        float max = 0f;
        int index = 0;
        for (int i=0; i<storage_list.size(); i++){
            long startTime = System.currentTimeMillis();   //获取开始时间
            float sim = MxNetUtils.calCosineSimilarity(source_emb, storage_list.get(i));
            long endTime = System.currentTimeMillis(); //获取结束时间
            System.out.println("人脸对比耗时： "+(endTime - startTime)+"ms");
            if (sim > max) {
                max = sim;
                index = i;
            }
        }

        return new Pair<>(index, max);
    }

    // singleton for the easy access
    private static Search search;
    public static Search getInstance() {
        if (search == null) {
            search = new Search();
        }
        return search;
    }
}
