import re


class HumanAnswerExtractor:
    """
    Extract the sections within human-written business insight text.
    The sections are:
    - Observations
    - Interpretations
    - Recommendations
    """

    @staticmethod
    def _extract_section(human_text: str, section: str) -> str:
        """
        Extracts the content of a given markdown section title from human-written business insight text.

        Args:
            human_text(str): human-written business insight text
            section(str): expected section to extract. Valid section names: 'observations', 'interpretations' 'recommendations'.

        Returns:
            str: text from the selected sections.
        """
        import re

        section = section.strip().lower()
        valid_sections = {"observations", "interpretations", "recommendations"}
        if section not in valid_sections:
            raise ValueError(f"Invalid section name '{section}'. Must be one of {valid_sections}")

        # Create a regex pattern to match the section content
        # It matches from the given section header up to the next ## header or end of text
        pattern = rf"^##\s*{section}\s*(.*?)(?=^##\s|\Z)"
        if section == "recommendations":
            # drop look-ahead ##
            pattern = rf"(?ms)^##\s*{section}\b\s*(.*)$"
        match = re.search(pattern, human_text, flags=re.S | re.I | re.M)

        return match.group(1).strip() if match else ""

    @staticmethod
    def extract_observations(human_text: str):
        return HumanAnswerExtractor._extract_section(human_text, "observations")

    @staticmethod
    def extract_interpretations(human_text: str):
        return HumanAnswerExtractor._extract_section(human_text, "interpretations")

    @staticmethod
    def extract_recommendations(human_text: str):
        return HumanAnswerExtractor._extract_section(human_text, "recommendations")
