import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAI, ChatOpenAI
from utils import PROMPT_1, verify_column_names, verify_num_columns
from tqdm import tqdm

st.set_page_config(layout='wide')

st.title("LLM - JabeLab")



dic_emp = {'OpenAI': ['gpt-4',
                      'gpt-4o',
                      'gpt-3.5-turbo-0125'],
           'Anthropic': ["claude-3-5-haiku-20241022",
                         "claude-3-5-sonnet-20241022",
                         "claude-3-opus-20240229"]}


empresa = st.selectbox("Seleccione una empresa: ", dic_emp.keys())
llm = st.selectbox("Seleccione el LLM", dic_emp[empresa])
api_key = st.text_input("API Key: ")
if api_key:
    if empresa == 'OpenAI':
        llm_oai = ChatOpenAI(
            model=llm,
            api_key=api_key
        )

    else:
        llm = ChatAnthropic(
            model=llm,
            api_key=api_key
        )

# ingresar documento en un formato específico
st.header("Carga de documentos")
st.write("Asegúrese de cargar un documento con la extensión '.csv' que se vea de la siguiente manera:")
prueba = pd.DataFrame({'id':['iji3h', '8uje3'],
              'texto': ['considero que no es apropiado pensar fuera...',
                        'correcto, debe tener otro...']})
st.dataframe(prueba)
st.write("El documento no puede pesar más de 200mb. Podrá descargar un documento de tipo '.csv' como el siguiente: ")

res = pd.DataFrame([[0,0,0,1,1,0,0,1,0, "claude-3-opus-20240229_0.1", "iji3h"]
[1,0,0,0,0,1,0,0,0, "claude-3-opus-20240229_0.1", '8uje3']], 
columns=["E",	"H", "MC", "PS", "F", "SN",	"TM", "R", "llm", "id"])

st.dataframe(res)

st.write("Las columnas hacen referencia a las categorías: EGOISM, HONESTY, MORAL-CONCERN, PROSOCIALITY, FAIRNESS, SOCIAL-NORMS, TAX-MORAL, RISKY")
with st.expander("Definiciones:"):
    st.write("""
    EGOISM: Egoism refers to behavior or attitudes driven by self-interest, prioritizing personal benefit over the welfare of others. In responses, egoism may manifest as a focus on maximizing one's own utility, wealth, or well-being without regard for collective or societal outcomes.
    HONESTY: Honesty involves truthfulness and integrity in communication and actions, often reflecting an internal commitment to moral or ethical principles. It can also relate to compliance with social or institutional rules.
    MORAL-CONCERN: Moral concern pertains to the consideration of ethical implications and the welfare of others when making decisions. It reflects an awareness of the moral consequences of one’s actions.
    PROSOCIALITY: Prosociality refers to voluntary actions intended to benefit others, such as helping, sharing, or cooperating. It reflects altruistic or community-oriented behavior.
    FAIRNESS: Fairness involves perceptions of justice, equity, and impartiality in interactions or resource distributions. It focuses on the idea that individuals should be treated equally or proportionately based on their contributions or needs.
    SOCIAL-NORMS: Social norms are the shared expectations or rules within a group or society regarding acceptable behavior. They guide individual actions based on what is deemed appropriate by others.
    TAX-MORAL: refers to the intrinsic motivation to comply with tax obligations, beyond legal enforcement. It reflects individuals' attitudes toward paying taxes as a civic duty or moral obligation.
    RISKY: pertains to behaviors or decisions involving uncertainty and the potential for loss. It often reflects a willingness to take chances for the possibility of higher rewards.
    """)

doc = st.file_uploader("Cargue el documento", ['csv'])

df = pd.read_csv(doc)

if not verify_column_names(df):
    st.error("El número o el nombre de las columnas no coinciden.")
    st.stop()

num_runs = st.slider("Seleccione un número de ejecuciones", 5, 25)
if st.button("Ejecutar"):
    infos = []

    for rep in range(num_runs):
        for temp in [0.1, 0.4, 0.8, 1]:
            for text, id in tqdm(zip(df['texto'].values,df['id'].values)):
                ans = llm.invoke(PROMPT_1 + " " + text, temperature=temp)
                ans = ans.content
                num = ans.find("{")
                ans = ans[num:]
                dictio = {'llm':f"{llm}_{temp}",
                                'info':ans,
                                'id': id,
                                'rep':rep}
                infos.append(dictio)
    series = []
    for dictio in infos:
        try:
            di = eval(dictio['info'])
            di['llm'] = dictio['llm']
            di['code'] = dictio['code']
            di['rep'] = dictio['rep']
            series.append(pd.Series(di))
        except:
            print("Error")

    sample = pd.concat(series, axis=1).T.iloc[::, :11]

    st.download_button(
        label="Descargar CSV",
        data=sample.to_csv(index=False).encode("utf-8"),
        file_name="results.csv",
        mime="text/csv",
    )



