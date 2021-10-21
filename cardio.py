import pickle

class Cardio(object):
    def __init__(self):
        self.weight_robust_scaler = pickle.load(open('model/rs_scaler_weight.pkl', 'rb'))
        self.bmi_robust_scaler = pickle.load(open('model/rs_scaler_bmi.pkl', 'rb'))
        self.ap_hi_robust_scaler = pickle.load(open('model/rs_scaler_ap_hi.pkl', 'rb'))
        self.ap_lo_robust_scaler = pickle.load(open('model/rs_scaler_ap_lo.pkl', 'rb'))

    def data_preparation(self, df):
        df1 = df.copy()
        df1['weight'] = self.weight_robust_scaler.transform(df1[['weight']].values)
        df1['bmi'] = self.bmi_robust_scaler.transform(df1[['bmi']].values)
        df1['ap_hi'] = self.ap_hi_robust_scaler.transform(df1[['ap_hi']].values)
        df1['ap_lo'] = self.ap_lo_robust_scaler.transform(df1[['ap_lo']].values)

        return df1
