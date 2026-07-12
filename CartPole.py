import taichi as ti
import math

#振子
#taichiの初期化
ti.init(arch=ti.cpu)

#パラメータの設定
g = 9.8 #重力加速度
l = 0.5 #ひもの長さの半分
dt = 0.02 #ステップ数
M = 1.0 #台車の質量
m = 0.1 #ボールの質量

# 変数の定義
theta = ti.field(dtype = ti.f32 ,shape=()) #ボールの角度
omega = ti.field(dtype = ti.f32,shape= ()) #ボールの角速度
x = ti.field(dtype = ti.f32,shape=())#台車の位置
v = ti.field(dtype = ti.f32,shape = ()) #台車の速度
force = ti.field(dtype = ti.f32,shape = ()) #台車に加える力

#カーネル
@ti.kernel
def init():
    x[None] =  0.5
    v[None] = 0.0
    theta[None] = 0.1
    omega[None] = 0.0

@ti.kernel
def update():
    #倒立振子の運動方程式
    total_mass = M + m

    temp = (force[None] + m * l * omega[None]**2 * ti.sin(theta[None]))/total_mass

    #ボールの加速度
    alpha = (g * ti.sin(theta[None]) - temp * ti.cos(theta[None])/(l *(4.0/3.0 - m * ti.cos(theta[None])**2)/total_mass))
    #台車の加速度
    acc = temp - m * l * alpha * ti.cos(theta[None])/total_mass

    #速度と位置の更新
    x[None] += v[None] * dt
    v[None] += acc * dt
    theta[None] += omega[None] * dt
    omega[None] += alpha * dt

#メインループ
init()
gui = ti.GUI("CartPole",res=(800,400))

while gui.running:
    # 1. キーボード入力の受け取り
    gui.get_event()
    if gui.is_pressed(ti.GUI.LEFT, 'a'):
        force[None] = -0.01  # 左に押す
    elif gui.is_pressed(ti.GUI.RIGHT, 'd'):
        force[None] = 0.01   # 右に押す
    else:
        force[None] = 0.0    # 何も押していない時は力を0にする
    update()

    #描画のための座標計算
    cart_x = x[None]
    cart_y = 0.2
    pole_angle= theta[None]

    # ポール先端の座標計算（描画用に長さを調整）
    draw_L = 0.3
    tip_x = cart_x + draw_L * math.sin(pole_angle)
    tip_y = cart_y + draw_L * math.cos(pole_angle)

    gui.clear(0x222222)
    
    # 地面（グレーの線）
    gui.line(begin=(0.0, 0.2), end=(1.0, 0.2), radius=2, color=0x888888)
    
    # 台車（太い線で四角形を表現）
    gui.line(begin=(cart_x - 0.05, cart_y), end=(cart_x + 0.05, cart_y), radius=12, color=0xAAAAAA)
    
    # ポール（白い線）
    gui.line(begin=(cart_x, cart_y), end=(tip_x, tip_y), radius=4, color=0xFFFFFF)
    
    # 支点（緑の円）
    gui.circle(pos=(cart_x, cart_y), radius=6, color=0x00FF00)

    gui.show()