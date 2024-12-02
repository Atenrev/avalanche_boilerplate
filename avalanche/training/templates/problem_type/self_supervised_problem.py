from avalanche.models import avalanche_forward

from avalanche.training.templates.strategy_mixin_protocol import (
    SelfSupervisedStrategyProtocol,
    TSGDExperienceType,
    TMBInput,
    TMBOutput,
)


class SelfSupervisedProblem(
    SelfSupervisedStrategyProtocol[TSGDExperienceType, TMBInput, TMBOutput]
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def mb_x(self):
        """Current mini-batch augmented input 1."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[0]

    @property
    def mb_y(self):
        """Current mini-batch target."""
        mbatch = self.mbatch
        assert mbatch is not None
        return mbatch[1]

    @property
    def mb_task_id(self):
        """Current mini-batch task labels."""
        mbatch = self.mbatch
        assert mbatch is not None
        assert len(mbatch) >= 3, "Task label not found."
        return mbatch[-1]

    def criterion(self):
        """Loss function for self-supervised problems."""
        if self.is_training or self._eval_criterion is None:
            return self._criterion(self.mb_output)
        else:
            return self._eval_criterion(self.mb_output, self.mb_y)

    def forward(self):
        """Compute the model's output given the current mini-batch."""
        def forward_single_input(input_data):
            if hasattr(self.experience, "task_labels"):
                return avalanche_forward(self.model, input_data, self.mb_task_id)
            return self.model(input_data)

        if isinstance(self.mb_x, (list, tuple)):
            return [forward_single_input(x) for x in self.mb_x]

        return forward_single_input(self.mb_x)

    def _unpack_minibatch(self):
        """Check if the current mini-batch has 3 components."""
        mbatch = self.mbatch
        assert mbatch is not None

        if isinstance(mbatch, tuple):
            mbatch = list(mbatch)
            self.mbatch = mbatch

        for i in range(len(mbatch)):
            mbatch[i] = mbatch[i].to(self.device, non_blocking=True)  # type: ignore


__all__ = ["SelfSupervisedProblem"]