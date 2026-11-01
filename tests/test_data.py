"""Test temporal boundaries, horizon alignment and feature causality."""
import unittest
import numpy as np
import pandas as pd
from src.data import Windows, calendar_features, TARGET, load_hourly

class WindowTests(unittest.TestCase):
    def test_exact_forecast_offsets(self):
        frame=pd.DataFrame({TARGET:np.arange(200,dtype=float)},index=pd.date_range("2016-01-01",periods=200,freq="h"))
        ds=Windows(frame,np.ones(200,dtype=bool),frame.mean()*0,frame.std()*0+1)
        x,y=ds[0]
        self.assertEqual(x.shape,(120,1))
        np.testing.assert_array_equal(x[:,0],np.arange(120))
        np.testing.assert_array_equal(y,[120,125,131,143])
        self.assertEqual(ds.origins[-1]+24,199)

    def test_missing_targets_excluded(self):
        frame=pd.DataFrame({TARGET:np.arange(200,dtype=float)},index=pd.date_range("2016-01-01",periods=200,freq="h"))
        observed=np.ones(200,dtype=bool); observed[143]=False
        ds=Windows(frame,observed,frame.mean()*0,frame.std()*0+1)
        self.assertNotIn(119,ds.origins)
        self.assertTrue(observed[ds.origins[:,None]+ds.horizons].all())

    def test_calendar_wrap(self):
        idx=pd.date_range("2016-01-01",periods=25,freq="h")
        f=calendar_features(idx)
        np.testing.assert_allclose(f[0,:2],f[24,:2],atol=1e-6)
        self.assertTrue(np.abs(f).max()<=1)

    def test_real_split_and_training_statistics(self):
        parts,observed,mean,std,audit=load_hourly()
        self.assertLess(parts["train"].index.max(),parts["validation"].index.min())
        self.assertLess(parts["validation"].index.max(),parts["test"].index.min())
        np.testing.assert_allclose(mean,parts["train"].mean())
        for name,frame in parts.items():
            self.assertTrue((frame.index[1:]-frame.index[:-1] == pd.Timedelta(hours=1)).all())
            ds=Windows(frame,observed[name],mean,std)
            self.assertGreaterEqual(ds.origins[0]-119,0)
            self.assertLess(ds.origins[-1]+24,len(frame))

if __name__=="__main__":
    unittest.main()
