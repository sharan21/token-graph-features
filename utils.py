def clean_line(line):
    line = line.strip('\n')
    clean = ''
    for c in line:
        if c.isalpha():
            clean += c
        else:
            clean += ' '
    return clean

