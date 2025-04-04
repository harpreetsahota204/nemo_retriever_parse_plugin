import os
import fiftyone as fo
from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types

from .nemo_retriever import run_nemo_retriever_parse

def _handle_calling(
        uri, 
        sample_collection, 
        api_key,
        delegate=False
        ):
    ctx = dict(dataset=sample_collection)

    params = dict(
        api_key=api_key,
        delegate=delegate
        )
    return foo.execute_operator(uri, ctx, params=params)

class NemoRetrieverParse(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            # The operator's URI: f"{plugin_name}/{name}"
            name="run_nemo_retriever_parse",  # required

            # The display name of the operator
            label="Parse Documents with NVIDIA NeMo Retriever Parse",  # required

            # A description for the operator
            description="Parse Documents with NVIDIA NeMo Retriever Parse",

            icon="/assets/icons8-nvidia.svg",
            )

    def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()

        inputs.str(
            "api_key",            
            required=True,
            label="Your NVIDIA API Key",
            description=(
                "NOTE: You'll need to set up an NVIDIA API key to use this plugin. You can get one by following this link: https://nvda.ws/3LspiUP"
                )
            )

        inputs.bool(
            "delegate",
            default=False,
            required=True,
            label="Delegate execution?",
            description=("If you choose to delegate this operation you must first have a delegated service running. "
            "You can launch a delegated service by running `fiftyone delegated launch` in your terminal"),
            view=types.CheckboxView(),
        )

        inputs.view_target(ctx)

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        """Implement this method if you want to programmatically *force*
        this operation to be delegated or executed immediately.

        Returns:
            whether the operation should be delegated (True), run
            immediately (False), or None to defer to
            `resolve_execution_options()` to specify the available options
        """
        return ctx.params.get("delegate", False)


    def execute(self, ctx):
        """Executes the actual operation based on the hydrated `ctx`.
        All operators must implement this method.

        This method can optionally be implemented as `async`.

        Returns:
            an optional dict of results values
        """
        view = ctx.target_view()
        api_key = ctx.params.get("api_key")
        bbox_field = ctx.params.get("bbox_field")
        output_field = ctx.params.get("output_field")
        confidence_threshold = ctx.params.get("confidence_threshold")
        
        run_nemo_retriever_parse(
            dataset=view,
            api_key = api_key,
            )
        

        ctx.ops.reload_dataset()

    def __call__(
            self, 
            sample_collection, 
            api_key,
            delegate=False
            ):
        return _handle_calling(
            self.uri,
            sample_collection,
            api_key,
            delegate=delegate
            )

def register(p):
    """Always implement this method and register() each operator that your
    plugin defines.
    """
    p.register(NemoRetrieverParse)