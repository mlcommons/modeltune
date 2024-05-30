from pydantic import BaseModel, Field
import json
from llama_recipes.inference.prompt_format_utils import SafetyCategory

from constants import (
    VIOLENT_CRIMES_NAME,
    VIOLENT_CRIMES_DESCRIPTION,
    NON_VIOLENT_CRIMES_NAME,
    NON_VIOLENT_CRIMES_DESCRIPTION,
    SEX_RELATED_CRIMES_NAME,
    SEX_RELATED_CRIMES_DESCRIPTION,
    CHILD_SEXUAL_EXPLOITATION_NAME,
    CHILD_SEXUAL_EXPLOITATION_DESCRIPTION,
    HATE_NAME,
    HATE_DESCRIPTION,
    SUICIDE_AND_SELF_HARM_NAME,
    SUICIDE_AND_SELF_HARM_DESCRIPTION,
    INDISCRIMINATE_WEAPONS_CBRENE_NAME,
    INDISCRIMINATE_WEAPONS_CBRENE_DESCRIPTION,
    SPECIALIZED_ADVICE_NAME,
    SPECIALIZED_ADVICE_DESCRIPTION,
    INTELLECTUAL_PROPERTY_NAME,
    INTELLECTUAL_PROPERTY_DESCRIPTION,
    UNKNOWN_HAZARD_NAME,
    UNKNOWN_HAZARD_DESCRIPTION,
)


class TaxonomyCategory(BaseModel):
    name: str
    description: str = ""
    value: bool = False

    def __hash__(self):
        return hash(self.name)

    def to_llama_safety_category(self):
        return SafetyCategory(name=self.name, description=self.description)


class MLCTaxonomyCategories(BaseModel):
    violent_crimes: TaxonomyCategory = Field(
        default_factory=lambda: TaxonomyCategory(
            name=VIOLENT_CRIMES_NAME, description=VIOLENT_CRIMES_DESCRIPTION
        )
    )

    non_violent_crimes: TaxonomyCategory = Field(
        default_factory=lambda: TaxonomyCategory(
            name=NON_VIOLENT_CRIMES_NAME, description=NON_VIOLENT_CRIMES_DESCRIPTION
        )
    )

    sex_related_crimes: TaxonomyCategory = Field(
        default_factory=lambda: TaxonomyCategory(
            name=SEX_RELATED_CRIMES_NAME, description=SEX_RELATED_CRIMES_DESCRIPTION
        )
    )

    child_sexual_exploitation: TaxonomyCategory = Field(
        default_factory=lambda: TaxonomyCategory(
            name=CHILD_SEXUAL_EXPLOITATION_NAME,
            description=CHILD_SEXUAL_EXPLOITATION_DESCRIPTION,
        )
    )

    hate: TaxonomyCategory = Field(
        default_factory=lambda: TaxonomyCategory(
            name=HATE_NAME, description=HATE_DESCRIPTION
        )
    )

    suicide_and_self_harm: TaxonomyCategory = Field(
        default_factory=lambda: TaxonomyCategory(
            name=SUICIDE_AND_SELF_HARM_NAME,
            description=SUICIDE_AND_SELF_HARM_DESCRIPTION,
        )
    )

    indiscriminate_weapons_cbrne: TaxonomyCategory = Field(
        default_factory=lambda: TaxonomyCategory(
            name=INDISCRIMINATE_WEAPONS_CBRENE_NAME,
            description=INDISCRIMINATE_WEAPONS_CBRENE_DESCRIPTION,
        )
    )

    specialized_advice: TaxonomyCategory = Field(
        default_factory=lambda: TaxonomyCategory(
            name=SPECIALIZED_ADVICE_NAME, description=SPECIALIZED_ADVICE_DESCRIPTION
        )
    )

    intellectual_property: TaxonomyCategory = Field(
        default_factory=lambda: TaxonomyCategory(
            name=INTELLECTUAL_PROPERTY_NAME,
            description=INTELLECTUAL_PROPERTY_DESCRIPTION,
        )
    )

    unknown_hazard: TaxonomyCategory = Field(
        default_factory=lambda: TaxonomyCategory(
            name=UNKNOWN_HAZARD_NAME, description=UNKNOWN_HAZARD_DESCRIPTION
        )
    )

    def model_dump_json(self):
        """Json string with key being snake case category name and value being the bool. e.g. hazard_name: True"""
        return json.dumps(
            {
                k: v.value
                for k, v in self.__dict__.items()
                if isinstance(v, TaxonomyCategory)
            }
        )

    def model_dump(self):
        """Dict with key being snake case category name and value being the bool. e.g. hazard_name: True"""
        return {
            k: v.value
            for k, v in self.__dict__.items()
            if isinstance(v, TaxonomyCategory)
        }

    @classmethod
    def from_category_list(cls, category_list: list[str]):
        mlc_tax_categories = cls()
        for category_name in category_list:
            if hasattr(mlc_tax_categories, category_name):
                setattr(
                    mlc_tax_categories,
                    category_name,
                    TaxonomyCategory(name=category_name, value=True),
                )
            else:
                raise ValueError(f"Invalid category name: {category_name}")
        return mlc_tax_categories
