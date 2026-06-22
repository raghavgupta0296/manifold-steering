from dataclasses import dataclass


@dataclass(frozen=True)
class WeekdayExample:
    entity: str
    number_word: str
    number_int: int
    answer: str
    raw_input: str
    raw_output: str


class WeekdayData:
    """
    Recreates the repo's weekdays natural-domain arithmetic data.

    Prompt:
        Q: What day is {number} days after {entity}?
        A:

    Output:
        " " + answer

    Example:
        Q: What day is three days after Thursday?
        A:

        -> " Sunday"
    """

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
    }

    template = "Q: What day is {number} days after {entity}?\nA:"
    output_prefix = " "

    def __init__(self):
        self.examples = self._build_examples()

    def _build_examples(self) -> list[WeekdayExample]:
        examples = []

        for entity_index, entity in enumerate(self.days):
            for number_word, number_int in self.numbers.items():
                answer_index = (entity_index + number_int) % len(self.days)
                answer = self.days[answer_index]

                raw_input = self.template.format(
                    number=number_word,
                    entity=entity,
                )
                raw_output = self.output_prefix + answer

                examples.append(
                    WeekdayExample(
                        entity=entity,
                        number_word=number_word,
                        number_int=number_int,
                        answer=answer,
                        raw_input=raw_input,
                        raw_output=raw_output,
                    )
                )

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> WeekdayExample:
        return self.examples[idx]

    def as_list(self) -> list[WeekdayExample]:
        return self.examples


if __name__ == "__main__":
    data = WeekdayData()

    print(f"Number of examples: {len(data)}")
    print()

    for example in data[:2]:
        print(example.raw_input + example.raw_output)
        print()
