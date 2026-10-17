from novexa.document_engine import split_passages, retrieve
def test_retrieve():
    p=split_passages([{"page":1,"text":"Neural networks learn patterns from data.\n\nThey use layers."},{"page":2,"text":"The ocean contains salt water."}])
    r=retrieve(p,"neural networks patterns")
    assert r and r[0]["page"]==1
