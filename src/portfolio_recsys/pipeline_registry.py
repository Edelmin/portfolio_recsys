"""Module description"""

from .pipelines.extraction_data_from_html.pipeline import (
    create_pipeline as extraction_data_from_html_gen,
)

from .pipelines.stratified_random_sampling.pipeline import (
    create_pipeline as stratified_random_sampling_gen,
)

extraction_data_from_html = extraction_data_from_html_gen()
stratified_random_sampling = stratified_random_sampling_gen()


def register_pipelines():
    return {
        "__default__": extraction_data_from_html + stratified_random_sampling,
        "extraction_data_from_html": extraction_data_from_html,
        "stratified_random_sample": stratified_random_sampling,
    }
