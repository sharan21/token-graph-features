def clean_line(line, bad=[',','.', ';', '(', ')', '/', '`', '%', '"', '-', '\\','\'',]):
    line = line.strip('\n')
    clean = ''
    for c in line:
        if c not in bad:
            clean += c
    return clean

