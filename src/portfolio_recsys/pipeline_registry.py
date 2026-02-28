from .etl_extraction.pipeline import create_pipeline as etl_extraction_test

etl_extraction = etl_extraction_test()


def register_pipelines():
    return {"__default__": etl_extraction}
