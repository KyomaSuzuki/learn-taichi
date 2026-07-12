import taichi as ti
import math

#振子
#taichiの初期化
ti.init(arch=ti.cpu)

#パラメータの設定
g = 9.8
l = 0.3
dt = 0.01

# 変数の定義
theta = ti.field(dtype = ti.f32 ,shape=()) #ボールの角度
omega = ti.field(dtype = ti.f32,shape= ()) #ボールの角速度

#カーネル
@ti.kernel
def init():
    theta[None] = 0.1
    omega[None] = 0.0

@ti.kernel
def update():
    alpha = (g/l) * ti.sin(theta[None])

    omega[None] += alpha * dt
    theta[None] += omega[None] * dt

#メインループ
init()
gui = ti.GUI("Inverted Pendulum",res=(600,600))

while gui.running:
    update()

    #描画のための座標計算
    current_theta = theta[None]

    base_x,base_y = 0.5,0.5

    tip_x = base_x + l * math.sin(current_theta)
    tip_y = base_y + l * math.cos(current_theta)

    gui.clear(0x222222)

    gui.line(begin=(base_x,base_y),end=(tip_x,tip_y),radius = 5,color = 0xFFFFFF)
    gui.circle(pos=(base_x,base_y),radius = 10,color =0x00FF00)
    gui.circle(pos=(tip_x,tip_y),radius = 15,color = 0x4488FF)
    gui.show()
    
