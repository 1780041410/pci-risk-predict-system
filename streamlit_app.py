import streamlit as st
import numpy as np
import warnings
import shap
import matplotlib.pyplot as plt
import pickle 
warnings.filterwarnings('ignore')
data_dict = {'Sex': 1,
 'Age': 65,
 'BMI': 23.2,
 'Weight': 72,
 'History of hypertension': 0,
 'History of diabetes': 1,
 'History of smoking': 1,
 'History of coronary heart disease': 0,
 'History of  myocardial infarction': 0,
 'History of PCI': 0,
 'History of CABG': 0,
 'History of cerebral infarction': 1,
 'History of cerebral hemorrhage': 0,
 'History of peptic ulcer': 1,
 'History of peripheral vascular diseases': 0,
 'History of renal insufficiency': 0,
 'History of heart failure': 0,
 'History of atrial fibrillation': 0,
 'Classification of symptoms': 3,
 'Systolic blood pressure': 110,
 'Diastolic blood pressure': 79,
 'Heart rate': 80,
 'Hgb': 153,
 'PLT': 150,
 'TC': 3.44,
 'TG': 2.18,
 'LDL-C': 2.22,
 'HDL-C': 0.81,
 'Cr': 53,
 'Urea': 6.23,
 'ALB': 40.4,
 'PT': 12.1,
 'APTT': 26.1,
 'INR': 1.06,
 'ALT': 27.7,
 'AST': 156.3,
 'CCr': 124.56,
 'FBG': 4.53,
 'WBC': 9.73,
 'Hs-cTNT': 2464,
 'Position of stents': 2,
 'Bare stent': 0,
 'Drug-eluting stent': 1.0,
 'DCB': 0.0,
 'Number of stents': 1,
 'aspirin and ticagrelor': 0.0,
 'aspirin and clopidogrel': 1.0,
 'ACEI or ARB': 0.0,
 'β-blockers': 0.0,
 'Statin': 1.0,
 'Warfarin': 0.0,
 'PPI': 1.0,
 'Other new anticoagulants': 0.0}
features_name = ['Sex', 'Age', 'BMI', 'Weight', 'History of hypertension',
                 'History of diabetes', 'History of smoking', 'History of coronary heart disease',
                 'History of  myocardial infarction', 'History of PCI', 'History of CABG',
                 'History of cerebral infarction', 'History of cerebral hemorrhage',
                 'History of peptic ulcer', 'History of peripheral vascular diseases',
                 'History of renal insufficiency', 'History of heart failure',
                 'History of atrial fibrillation', 'Classification of symptoms',
                 'Systolic blood pressure', 'Diastolic blood pressure', 'Heart rate',
                 'Hgb', 'PLT', 'TC', 'TG', 'LDL-C', 'HDL-C', 'Cr', 'Urea', 'ALB', 'PT',
                 'APTT', 'INR', 'ALT', 'AST', 'CCr', 'FBG', 'WBC', 'Hs-cTNT',
                 'Position of stents', 'Bare stent', 'Drug-eluting stent', 'DCB',
                 'Number of stents', 'aspirin and ticagrelor', 'aspirin and clopidogrel',
                 'ACEI or ARB', 'β-blockers', 'Statin', 'Warfarin', 'PPI', 'Other new anticoagulants']

 #load the predict model
hemorrhage_model = pickle.load(open('hemorrhage.pkl', 'rb'))
cardiogenic_death_model = pickle.load(open('./cardiogenic_death.pkl', 'rb'))
stent_restenosis_model = pickle.load(open('./stent_restenosis.pkl', 'rb'))
model_name = ["Hemorrhoea","Cardiac death","In-sent restenosis"]
model_list = [hemorrhage_model,cardiogenic_death_model,stent_restenosis_model]
with st.form("my_form"):
    Sex = st.selectbox('Sex',(0,1), index =1)
    Age = st.number_input('Age',value=65)
    BMI = st.number_input('BMI',value=23.2)
    Weight = st.number_input('Weight',value=72)
    Hypertension = st.selectbox('History of hypertension',(0,1), index =0)
    Diabetes = st.selectbox('History of diabetes',(0,1), index = 1)
    Smoke_history = st.selectbox('History of smoking',(0,1), index = 1)
    Family_history_CVD = st.selectbox('History of coronary heart disease',(0,1), index = 0)
    Prior_myocardial_infarction = st.selectbox('History of  myocardial infarction',(0,1), index = 0)
    History_of_PCI = st.selectbox('History of PCI',(0,1), index = 0)
    History_of_CABG = st.selectbox('History of CABG',(0,1), index = 0)
    History_of_cerebral_infarction = st.selectbox('History of cerebral infarction',(0,1), index = 1)
    History_of_cerebral_hemorrhage = st.selectbox('History of cerebral hemorrhage',(0,1), index = 0)
    Peptic_ulcer = st.selectbox('History of peptic ulcer',(0,1), index = 1 )
    Peripheral_vascular_disease = st.selectbox('History of peripheral vascular diseases',(0,1), index = 0)
    Renal_insufficiency = st.selectbox('History of renal insufficiency',(0,1), index = 0)
    History_of_heart_failure = st.selectbox('History of heart failure',(0,1), index = 0)
    History_of_atrial_fibrillation = st.selectbox('History of atrial fibrillation',(0,1), index = 0)
    clinical_manifestations = st.number_input('Classification of symptoms',value=3)
    Systolic_blood_pressure = st.number_input('Systolic blood pressure',value=110)
    Diastolic_pressure = st.number_input('Diastolic blood pressure',value=79)
    Heart_rate = st.number_input('Heart rate',value=80)
    Hgb = st.number_input('Hgb',value=153)
    PLT = st.number_input('PLT',value=150)
    TC = st.number_input('TC',value=3.44)
    TG = st.number_input('TG',value=2.18)
    LDL_C = st.number_input('LDL-C',value=2.22)
    HDL_C = st.number_input('HDL-C',value=0.81)
    Cr = st.number_input('Cr',value=53)
    Urea = st.number_input('Urea',value=6.23)
    ALB = st.number_input('ALB',value=40.4)
    PT = st.number_input('PT',value=12.1)
    APTT = st.number_input('APTT',value=26.1)
    INR = st.number_input('INR',value=1.06)
    ALT = st.number_input('ALT',value=27.7)
    AST = st.number_input('AST',value=156.3)
    CCr = st.number_input('CCr',value=124.56)
    Fasting_blood_sugar = st.number_input('FBG',value=4.53)
    WBC = st.number_input('WBC',value=9.73)
    hsTNT = st.number_input('Hs-cTNT',value=2464)
    Stent_implanted_blood_vessel = st.number_input('Position of stents',value=2)
    Bare_stent = st.selectbox('Bare stent',(0,1), index = 0)
    Drug_eluting_stent = st.selectbox('Drug-eluting stent',(0,1), index =1)
    DCB = st.selectbox('DCB',(0,1), index =0)
    Total_number_of_stents_implanted = st.number_input('Number of stents',value=1)
    Bayaspirin = st.selectbox('aspirin and ticagrelor',(0,1), index =0)
    Plavix = st.selectbox('aspirin and clopidogrel',(0,1), index =1)
    ACEI_or_ARB = st.selectbox('ACEI or ARB',(0,1), index =0)
    B_blockers = st.selectbox('β-blockers',(0,1), index =0)
    Statin = st.selectbox('Statin',(0,1), index =0)
    Warfarin = st.selectbox('Warfarin',(0,1), index =0)
    PPI = st.selectbox('PPI',(0,1), index =1)
    Other_new_anticoagulants = st.selectbox('Other new anticoagulants',(0,1), index =0)
    submitted = st.form_submit_button("Predict")
    if submitted:
        x_test = np.array([[Sex, Age, BMI, Weight,Hypertension,Diabetes,Smoke_history,Family_history_CVD,Prior_myocardial_infarction,History_of_PCI,History_of_CABG,History_of_cerebral_infarction,
        History_of_cerebral_hemorrhage,Peptic_ulcer,Peripheral_vascular_disease,Renal_insufficiency,History_of_heart_failure,History_of_atrial_fibrillation,clinical_manifestations,Systolic_blood_pressure,
        Diastolic_pressure,Heart_rate,Hgb,PLT,TC,TG,LDL_C,HDL_C,Cr,Urea,ALB,PT,APTT,INR,ALT,AST,CCr,Fasting_blood_sugar,WBC,hsTNT,Stent_implanted_blood_vessel,Bare_stent,
        Drug_eluting_stent,DCB,Total_number_of_stents_implanted,Bayaspirin,Plavix,ACEI_or_ARB,B_blockers,Statin,Warfarin,PPI,Other_new_anticoagulants]])
        for index ,(name,model) in enumerate(zip(model_name,model_list)):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_test) 
            temp = np.round(x_test, 2)
            shap.force_plot(explainer.expected_value[1], shap_values[1],temp, matplotlib=True, show=False,feature_names=features_name)
            plt.xticks(fontproperties='Times New Roman', size=10)
            plt.yticks(fontproperties='Times New Roman', size=10)
            plt.tight_layout()
            plt.savefig("PCI force plot.png",dpi=700) 
            pred = model.predict_proba(x_test)
            st.markdown("#### ({}).Based on feature values, predicted possibility of {} is {}%".format(index+1,name,round(pred[0][1], 4)*100))
            st.image('PCI force plot.png')
