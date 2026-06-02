import zipfile
import pickletools
import io

def find_all_globals(ckpt_path):
    globals_found = set()
    
    with zipfile.ZipFile(ckpt_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".pkl") or "data.pkl" in name or name.endswith("/data"):
                with zf.open(name) as f:
                    data = f.read()
                try:
                    out = io.StringIO()
                    pickletools.dis(io.BytesIO(data), out)
                    for line in out.getvalue().splitlines():
                        if "GLOBAL" in line or "STACK_GLOBAL" in line:
                            globals_found.add(line.strip())
                except Exception as e:
                    print(f"Skipping {name}: {e}")
    
    return globals_found

for g in sorted(find_all_globals("models/Clasp_2D/model_checkpoints/t1post_and_t2.ckpt")):
    print(g)