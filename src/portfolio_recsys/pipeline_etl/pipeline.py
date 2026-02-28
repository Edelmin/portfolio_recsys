from kedro.pipeline import Pipeline, node, pipeline
from .nodes import foo


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [node(func=foo, inputs=None, outputs="dummy_output", name="dummy_function")],
        tags=["etl"],
    )
