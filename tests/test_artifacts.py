import pandas as pd
from novexa.artifact_generator import dataframe_xlsx, text_docx, text_pdf
def test_artifacts():
    assert dataframe_xlsx(pd.DataFrame({"x":[1]})).startswith(b"PK")
    assert text_docx("Title","Hello").startswith(b"PK")
    assert text_pdf("Title","Hello").startswith(b"%PDF")
