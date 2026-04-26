def test_langchain_imports():
    import langchain
    import langgraph
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import BaseMessage
    from langgraph.graph import StateGraph
    assert True

def test_protobuf_version():
    import google.protobuf
    version = tuple(int(x) for x in google.protobuf.__version__.split('.'))
    assert version[0] < 4, f"protobuf {google.protobuf.__version__} >= 4.0, conflicts with existing constraint"
