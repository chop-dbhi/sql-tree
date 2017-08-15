# SQL Tree

Utility to generate the tree of relationships from a root table in a relational schema. It relies on foreign key constraints in the schema to determine paths.

## Install

Dependencies:

- PyYAML
- Graphviz

```
pip install -r requirements.txt
```

## Usage

Export a schema using the [sql-schema-exporter](https://github.com/chop-dbhi/sql-schema-exporter). This along with a root table to build the tree from are required.

```
python tree.py schema.json --table=some_table > some_table_tree.json
```

The output is a map to all tables by name and the path of join expressions. The list will be empty if a table is unreachable by the root.

```js
{
  "production_audiogramresult": [
    {
      "predicate": [
        {
          "operator": "=",
          "left_column": "id",
          "right_column": "patient_id"
        }
      ],
      "nullable": true,
      "reverse": true,
      "right_table": "production_audiogramtest",
      "left_table": "production_patient"
    },
    {
      "predicate": [
        {
          "operator": "=",
          "left_column": "id",
          "right_column": "test_id"
        }
      ],
      "nullable": true,
      "reverse": false,
      "right_table": "production_audiogramresult",
      "left_table": "production_audiogramtest"
    }
  ],

  // ...
}
```

This provides a data structure that contains the necessary information for joins to be expressed in a SQL query.

## Visual

The data structure is useful to work with, but not a good visual. Simply add the `--graphviz=output.pdf` option to generate PDF with the rendered tree. The filename and extension can be anything graphviz supports natively. Below is a snippet of the what the output may look like.

![tree visual](./example.png)


