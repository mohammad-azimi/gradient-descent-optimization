import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from PIL import Image


EQUATIONS = {
    "f1": r"f_1(x,y)=x^2+2y^2+e^{-x-y}-x+2y",
    "f2": r"f_2(x,y)=x^2+2y^2+\cos(x+y)-x+2y",
    "grad1": r"\nabla f_1=\left(\genfrac{}{}{0}{}{2x-e^{-x-y}-1}{4y-e^{-x-y}+2}\right)",
    "hess1": r"H_1=\left(\genfrac{}{}{0}{}{2+t\quad t}{t\quad 4+t}\right),\quad t=e^{-x-y}>0",
    "grad2": r"\nabla f_2=\left(\genfrac{}{}{0}{}{2x-\sin(x+y)-1}{4y-\sin(x+y)+2}\right)",
    "hess2": r"H_2=\left(\genfrac{}{}{0}{}{2-q\quad -q}{-q\quad 4-q}\right),\quad q=\cos(x+y)",
    "bound1": r"z^T H_1z=2z_1^2+4z_2^2+t(z_1+z_2)^2\geq 2\Vert z\Vert^2",
    "bound2": r"\left(\genfrac{}{}{0}{}{1\quad -1}{-1\quad 3}\right)\preceq H_2\preceq\left(\genfrac{}{}{0}{}{3\quad 1}{1\quad 5}\right)",
    "constants": r"\mu_1=2,\quad\mu_2=2-\sqrt{2}>0,\quad L_2=4+\sqrt{2}",
    "coercive": r"f_i(x,y)\geq(x-\frac{1}{2})^2+2(y+\frac{1}{2})^2-C_i,\quad C_1=\frac{3}{4},\ C_2=\frac{7}{4}",
    "ref1": r"x^*=\frac{1+t}{2},\quad y^*=\frac{t-2}{4},\quad t=e^{-3t/4}",
    "scalar1": r"\log t+\frac{3}{4}t=0,\quad t=\frac{4}{3} W(\frac{3}{4})",
    "ref2": r"s=x+y=\frac{3}{4}\sin s\quad\Rightarrow\quad s=0,\quad (x^*,y^*)=(\frac{1}{2},-\frac{1}{2})",
    "update": r"g_k=\nabla f(x_k),\quad d_k=-g_k,\quad x_{k+1}=x_k+\alpha_kd_k",
    "armijo": r"f(x_k-\alpha_kg_k)-f(x_k)\leq-\varepsilon\alpha_k\Vert g_k\Vert^2",
    "stop": r"\Vert\nabla f(x_k)\Vert<\delta",
    "golden": r"\alpha_k\approx\arg\min_{0<\alpha<1}\phi_k(\alpha),\quad\phi_k(\alpha)=f(x_k-\alpha g_k)",
    "ratio": r"\rho=\frac{\sqrt{5}-1}{2},\quad c=b-\rho(b-a),\quad d=a+\rho(b-a)",
    "descent": r"\nabla f(x_k)^Td_k=-\Vert\nabla f(x_k)\Vert^2<0\quad(g_k\ne 0)",
    "normal": r"f(\gamma(u))=c\quad\Rightarrow\quad\nabla f(\gamma(u))^T\gamma'(u)=0",
    "orthogonal": r"\phi_k'(\alpha_k)=g_{k+1}^Td_k=-g_{k+1}^Tg_k=0",
    "cosine": r"\cos\theta_k=\frac{g_{k+1}^Tg_k}{\Vert g_{k+1}\Vert\,\Vert g_k\Vert}",
    "error": r"\Vert x-x^*\Vert\leq\frac{\Vert\nabla f(x)\Vert}{\mu},\quad f(x)-f^*\leq\frac{\Vert\nabla f(x)\Vert^2}{2\mu}",
    "change1": r"\Delta f_1=g^Th+h_x^2+2h_y^2+t(e^{-r}-1+r),\quad r=h_x+h_y",
    "change2": r"\Delta f_2=g^Th+h_x^2+2h_y^2-2\cos s\sin^2(r/2)+\sin s(r-\sin r)",
}


def render_equations(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, latex in EQUATIONS.items():
        file = destination/f"{name}.png"
        math_to_image("$"+latex+"$", file, prop=FontProperties(size=12), dpi=350, color="black")
        with Image.open(file) as image:
            width = image.width/350.0
        manifest[name] = {"width_inches": min(width, 6.7)}
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    render_equations(Path(__file__).resolve().parent/"report"/"assets"/"equations")
