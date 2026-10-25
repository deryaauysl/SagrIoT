import sys, pandas as pd
from numbers_parser import Document
src, out = sys.argv[1], sys.argv[2]
doc = Document(src)
for si, sh in enumerate(doc.sheets):
    for ti, t in enumerate(sh.tables):
        rows = t.rows(values_only=True)
        print(f"== sheet {si} '{sh.name}' table {ti} '{t.name}': {t.num_rows} rows x {t.num_cols} cols")
        df = pd.DataFrame(rows[1:], columns=rows[0])
        df.to_csv(f"{out}_s{si}_t{ti}.csv", index=False)
        print(df.head(3).to_string()[:3000])
