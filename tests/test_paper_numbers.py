from experiments.figures import paper_numbers as pn


def test_formatting_helpers():
    assert pn.u3(0.00226) == r"$+2.26\times10^{-3}$" and pn.u3(-0.00008) == r"$-0.08\times10^{-3}$"
    assert pn.u3ci(-0.0004, 0.0003) == r"$[-0.40,+0.30]\times10^{-3}$"
    assert pn.pv(0.0004) == "$p<10^{-3}$" and pn.pv(0.1034) == "$p=0.103$"
    assert pn.th(13458) == "13{,}458" and pn.rel(1.576) == r"$+$158\%"


def test_section_map_names_the_section_of_each_macro():
    tex = "\\begin{abstract}\n\\numA\n\\end{abstract}\n\\section{Results}\n\\subsection{Text}\nx \\numB y \\numA\n"
    used = pn.section_map(tex)
    assert used["A"] == ["Abstract", "Text"] and used["B"] == ["Text"]
