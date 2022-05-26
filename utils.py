def clean_line(line, bad=[',','.', ';', '(', ')', '/', '`', '%', '"', '-', '\\','\'',]):
    clean = ''
    for c in line:
        if c not in bad:
            clean += c
    return clean

