from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from data.weekdays import WeekdayData
from model.llama import LlamaModel
from report_html import REPORT_PATH, append_or_replace_section


def main() -> None:
    data = WeekdayData()
    model = LlamaModel(LlamaModel.paper_model_name).load()

    examples = data.as_list()
    prompts = [example.raw_input for example in examples]

    probabilities = model.gather_prediction_probabilities(
        prompts=prompts,
        answer_values=data.days,
        output_prefix=data.output_prefix,
        full_vocab_softmax=True,
        include_other=True,
        batch_size=8,
    )

    weekday_probabilities = probabilities[:, : len(data.days)]
    predicted_indices = weekday_probabilities.argmax(dim=-1).tolist()
    predicted_answers = [data.days[index] for index in predicted_indices]

    correct = [
        predicted_answer == example.answer
        for predicted_answer, example in zip(predicted_answers, examples)
    ]
    accuracy = sum(correct) / len(correct)

    print(f"Model: {model.model_name}")
    print(f"Examples: {len(examples)}")
    print(f"Accuracy: {sum(correct)}/{len(correct)} = {accuracy:.2%}")
    print()

    for example, predicted_answer, is_correct, probs in zip(
        examples[:10],
        predicted_answers[:10],
        correct[:10],
        probabilities[:10],
    ):
        expected_probability = probs[data.days.index(example.answer)].item()
        predicted_probability = probs[data.days.index(predicted_answer)].item()
        other_probability = probs[-1].item()

        print(example.raw_input + example.raw_output)
        print(f"predicted: {predicted_answer} ({predicted_probability:.4f})")
        print(f"expected probability: {expected_probability:.4f}")
        print(f"other probability: {other_probability:.4f}")
        print(f"correct: {is_correct}")
        print()

    mean_correct_probability = sum(
        probabilities[index, data.days.index(example.answer)].item()
        for index, example in enumerate(examples)
    ) / len(examples)
    other_probability = probabilities[:, -1].mean().item()
    rows = "\n".join(
        f"""
        <tr>
          <td><code>{example.raw_input.replace(chr(10), '<br>') + example.raw_output}</code></td>
          <td>{example.answer}</td>
          <td>{predicted_answer}</td>
          <td>{str(is_correct)}</td>
        </tr>
        """
        for example, predicted_answer, is_correct in zip(
            examples[:10],
            predicted_answers[:10],
            correct[:10],
        )
    )
    append_or_replace_section(
        "experiment-1-baseline",
        f"""
  <section id="experiment-1-baseline">
    <h2>1. Baseline Weekday Scoring</h2>
    <p>
      Model: <code>{model.model_name}</code><br>
      Examples: <code>{len(examples)}</code><br>
      Weekday argmax accuracy: <code>{sum(correct)}/{len(correct)} = {accuracy:.2%}</code><br>
      Mean correct weekday probability: <code>{mean_correct_probability:.2%}</code><br>
      Mean other probability: <code>{other_probability:.2%}</code>
    </p>
    <table>
      <thead>
        <tr><th>Prompt + Expected</th><th>Expected</th><th>Predicted</th><th>Correct</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
""",
    )
    print(f"Updated report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
