import os
from pathlib import Path

def find_ref(file, ref):
    f = open(file, encoding="utf8")
    l = f.readlines()

    count=0
    ref_line = []
    ref_content = []
    for line in l:
        if "#"+ str(ref) in line:
            ref_line.append(count)
            ref_content.append(line)
        count += 1
    return ref_line, ref_content

path = Path("./")
path = path.parent.parent.absolute()
path = path.parent.absolute()
path = path.parent.absolute()
repo_path = path



ref_line = []
ref_content = []
ref_file = []
# r=root, d=directories, f = files
for r, d, f in os.walk(repo_path):
    for file in f:
        if file.endswith(".py"):
            ref_line_aux, ref_content_aux = find_ref( os.path.join(r, file) , 'TODO')

            if ref_line_aux and ref_content_aux:
                ref_line.append(ref_line_aux)
                ref_content.append(ref_content_aux)
                ref_file.append(os.path.join(r, file))



for i, file in enumerate(ref_file):
    print(f'\nFILE: {file.split("neurasim")[1]}')
    for j, line in enumerate(ref_line[i]):
        print(f' |-line {line} >> {ref_content[i][j]}', end='')
