def remove_delimiter(text):
    text = text.replace('\n', '')
    return text

def remove_separator(text):
    text = text.replace('\r', '')
    return text

def remove_empty(text):
    text = text.strip()
    return text

def remove_two_spaces(text):
    text = text.replace("  ", " ")
    return text

def remove_three_spaces(text):
    text = text.replace("   ", " ")
    return text

def df_to_txt(df, out_file_path, text_column = "text", label_column = "class_label"):
    """
    using fixtures to test
    """
    f = open(out_file_path, "w")
    for ix, row in df.iterrows():
        text = row[text_column]
        label = str(row[label_column])
        f.write(text)
        f.write('\n')
        f.write(label)
        f.write('\n')
    f.close()
    print ("writing txt finished")