package cz.nakoncisveta.eyetracksample.eyedetect;

import android.content.Intent;
import android.os.Bundle;
import android.app.Activity;
import android.view.View;

public class StartActivity extends Activity {

    public void onStartClick(View v)
    {
        Intent myIntent = new Intent(StartActivity.this, MainActivity.class);
        startActivity(myIntent);
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start);
    }

}
