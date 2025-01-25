def append_llm_output(df: pd.DataFrame, payload: dict, result_col: str) -> pd.DataFrame:
    """
    Updates the original DataFrame by assigning generated topic labels to each snippet
    based on the index of the snippet.
    :param df: pd.DataFrame, dataframe with snippets that need topic labels.
    :param payload: dict, prompt payload
    :return: pd.DataFrame, validation set plus updated 'gpt_topic_label' column
    """
    for index, row in df.iterrows():
        payload["snippet"] = row['snippet']

        # Generate the output for the snippet
        output = generate_llm_output(payload)
        
        # Update the DataFrame directly at the current index
        df.at[index, result_col] = output

        # Buffer between calls
        time.sleep(3)
        
    return df