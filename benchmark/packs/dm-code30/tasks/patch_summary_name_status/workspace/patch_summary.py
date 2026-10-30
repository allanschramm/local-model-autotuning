def summarize_name_status(lines):
    summary = {"added": [], "modified": [], "deleted": [], "renamed": []}
    for line in lines:
        status, path = line.split("\t")
        if status == "A":
            summary["added"].append(path)
        elif status == "M":
            summary["modified"].append(path)
        elif status == "D":
            summary["deleted"].append(path)
    return summary
