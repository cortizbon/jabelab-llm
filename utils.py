PROMPT_1 = """You are a psychology-economics researcher specialized in qualitative research.
You've been working for a long time in reading answers from interviews and classifying them in some categories.
These categories goes as follows:

EGOISM: Egoism refers to behavior or attitudes driven by self-interest, prioritizing personal benefit over the welfare of others. In responses, egoism may manifest as a focus on maximizing one's own utility, wealth, or well-being without regard for collective or societal outcomes.
HONESTY: Honesty involves truthfulness and integrity in communication and actions, often reflecting an internal commitment to moral or ethical principles. It can also relate to compliance with social or institutional rules.
MORAL-CONCERN: Moral concern pertains to the consideration of ethical implications and the welfare of others when making decisions. It reflects an awareness of the moral consequences of one’s actions.
PROSOCIALITY: Prosociality refers to voluntary actions intended to benefit others, such as helping, sharing, or cooperating. It reflects altruistic or community-oriented behavior.
FAIRNESS: Fairness involves perceptions of justice, equity, and impartiality in interactions or resource distributions. It focuses on the idea that individuals should be treated equally or proportionately based on their contributions or needs.
SOCIAL-NORMS: Social norms are the shared expectations or rules within a group or society regarding acceptable behavior. They guide individual actions based on what is deemed appropriate by others.
TAX-MORAL: refers to the intrinsic motivation to comply with tax obligations, beyond legal enforcement. It reflects individuals' attitudes toward paying taxes as a civic duty or moral obligation.
RISKY: pertains to behaviors or decisions involving uncertainty and the potential for loss. It often reflects a willingness to take chances for the possibility of higher rewards.

Keep these definitions in mind. I'm going to provide an answer of an interview and I want you to,
regarding the definitions you already have, provide a 1 for every category that you can see in the answer and a 0 instead.

For example, if you consider that the answer has tax-moral, honesty and moral-concern, you assign 1 to each, and 0 to the others.

Please, provide the answer as a python dictionary without any additional text or characters like points, spaces or any other character. Be careful with the names of the categories without mispelling them.

The names of the categories in the dictionary should go as follows: E for EGOISM, H for HONESTY, MC for MORAL-CONCERN, PS for PROSOCIALITY, F for FAIRNESS, SN for SOCIAL-NORMS, TM for TAX-MORAL and R for RISKY. Be sure that every category is within ''.

The text is as follows: 
 """

def verify_column_names(df):
    if verify_num_columns(df):
        nombre_columnas = ['texto', 'id']
        cols = df.columns
        for col in cols:
            if col not in nombre_columnas:
                return False
        return True
    return False

def verify_num_columns(df):
    if len(df.columns) == 2:
        return True
    return False 

