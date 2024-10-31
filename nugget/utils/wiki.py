no_space_tokens = {' '+t for t in list('.!?,') + ['\'d', '\'ve', '\'re', '\'s']}


def clean_text(text):
    text = text.strip().replace('\n', ' ').strip()
    while '  ' in text:
        text = text.replace('  ', ' ')

    text = text.replace('@-@', '-').replace('@-@', '-').replace('@.@', '.').replace('@,@', ',')
    tokens = text.split()
    tokens = [tokens[0]] + [' '+t for t in tokens[1:]]
    tokens = [t if t not in no_space_tokens else t[1:] for t in tokens]

    return tokens


def iterate_articles(dataset, min_sec_length=8):
    sections = []
    lines = []
    was_empty = True

    def put_lines():
        nonlocal lines, sections
        n_tok = len(' '.join(lines).strip().split())
        if n_tok >= min_sec_length:
            sections.append(clean_text(' '.join(lines)))
        lines = []

    for line in dataset['text']:
        line = line.rstrip('\n')
        if line.strip() == '':
            was_empty = True
            continue

        if was_empty:
            if line.startswith(' = = ') and line.endswith(' = = '):
                put_lines()
                continue
            elif line.startswith(' = ') and line.endswith(' = '):
                put_lines()
                if len(sections) > 0:
                    yield sections
                    lines, sections = list(), list()
                continue

        lines.append(line.strip())
        was_empty = False

    put_lines()
    if len(sections) > 0:
        yield sections
