import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk

from src.proceed import (
    SUPPORTED_EXTENSIONS,
    load_image,
    load_images_from_folder,
    save_image,
)
from src.metrics import compute_image_metrics, save_image_metrics_json
from src.run import ALGORITHMS, ALGORITHM_PARAMS, timer


def _format_file_size(file_path: Path | None) -> str:
    """Форматує розмір файлу у KB або MB. Якщо даних немає — повертає '-'."""
    if file_path is None or not file_path.is_file():
        return "-"
    size_bytes = file_path.stat().st_size
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / 1024:.1f} KB"


def _format_proc_time(seconds: float | None) -> str:
    """Форматує час обробки у ms (якщо < 1 с) або s. Якщо даних немає — повертає '-'."""
    if seconds is None:
        return "-"
    if seconds < 1.0:
        return f"{seconds * 1000.0:.1f} ms"
    return f"{seconds:.3f} s"


def _format_relative_path(file_path: Path | None) -> str:
    """Повертає відносний шлях збереженого файлу (не абсолютний). Якщо даних немає — повертає '-'."""
    if file_path is None:
        return "-"
    try:
        return os.path.relpath(file_path, Path.cwd())
    except ValueError:
        return file_path.name


class SpaceAnalyzeApp:
    """Графічний інтерфейс (Tkinter) для обробки та аналізу зображень космічних об'єктів."""

    def __init__(self, root: tk.Tk, output_dir: str | Path = r".\output") -> None:
        self.root = root
        self.root.title("Аналіз космічних об'єктів — Обробка зображень")
        self.root.geometry("1320x780")
        self.root.minsize(1020, 600)
        self.root.configure(bg="#f4f6f9")

        self.output_dir = Path(output_dir)

        # Список завантажених елементів: [(Image, Path), ...]
        self.loaded_items: list[tuple[Image.Image, Path]] = []
        # Словник результатів обробки за індексом:
        # {idx: {"img": Image, "saved_path": Path | None, "time": float}}
        self.processed_items: dict[int, dict] = {}
        self.current_index: int = 0
        self.is_processing: bool = False

        # Збереження поточних віджетів/значень параметрів обраного алгоритму
        self.param_getters: dict[str, callable] = {}

        # Збереження посилань на PhotoImage, щоб їх не видалив garbage collector
        self._orig_photo: ImageTk.PhotoImage | None = None
        self._proc_photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._rebuild_param_controls()
        self._update_info_bar()

    def _build_ui(self) -> None:
        # 1. Верхня панель завантаження та навігації по файлах (без progress bar)
        top_bar = tk.Frame(
            self.root,
            bg="#ffffff",
            padx=14,
            pady=8,
            highlightthickness=1,
            highlightbackground="#e2e8f0",
        )
        top_bar.pack(side=tk.TOP, fill=tk.X)

        self.btn_load_file = ttk.Button(
            top_bar,
            text="Завантажити зображення",
            command=self._on_load_file,
        )
        self.btn_load_file.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_load_folder = ttk.Button(
            top_bar,
            text="Завантажити папку",
            command=self._on_load_folder,
        )
        self.btn_load_folder.pack(side=tk.LEFT, padx=(0, 18))

        self.btn_prev = ttk.Button(
            top_bar,
            text="◀ Попереднє",
            command=self._on_prev_image,
            state=tk.DISABLED,
            width=13,
        )
        self.btn_prev.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_next = ttk.Button(
            top_bar,
            text="Наступне ▶",
            command=self._on_next_image,
            state=tk.DISABLED,
            width=13,
        )
        self.btn_next.pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(
            top_bar,
            text="Файл:",
            bg="#ffffff",
            fg="#475569",
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 6))

        self.file_var = tk.StringVar(value="Не обрано")
        self.file_combo = ttk.Combobox(
            top_bar,
            textvariable=self.file_var,
            state="disabled",
            width=45,
        )
        self.file_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.file_combo.bind("<<ComboboxSelected>>", self._on_select_file_combo)

        # 2. Довгий об'єднаний інфо-бар знизу вікна (фіксованої висоти, щоб ніколи не змінював розмір зон)
        self.bottom_info_bar = tk.Frame(
            self.root,
            bg="#1c2331",
            height=40,
            padx=18,
        )
        self.bottom_info_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.bottom_info_bar.pack_propagate(False)

        self.info_items_frame = tk.Frame(self.bottom_info_bar, bg="#1c2331")
        self.info_items_frame.pack(side=tk.LEFT, fill=tk.Y)

        # 3. Центральний контейнер: Ліва панель (Algorithm + параметри) + Права область зображень
        content_frame = tk.Frame(self.root, bg="#f4f6f9", padx=12, pady=10)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Ліва бічна панель керування алгоритмом ---
        self.sidebar = tk.Frame(
            content_frame,
            bg="#f8fafc",
            width=290,
            padx=16,
            pady=14,
            highlightthickness=1,
            highlightbackground="#e2e8f0",
        )
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        self.sidebar.pack_propagate(False)

        tk.Label(
            self.sidebar,
            text="Algorithm",
            bg="#f8fafc",
            fg="#1e293b",
            font=("Segoe UI", 12, "bold"),
            anchor="w",
        ).pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        tk.Frame(self.sidebar, bg="#e2e8f0", height=1).pack(side=tk.TOP, fill=tk.X, pady=(0, 12))

        tk.Label(
            self.sidebar,
            text="Choose Algorithm",
            bg="#f8fafc",
            fg="#475569",
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(side=tk.TOP, fill=tk.X, pady=(0, 4))

        algo_names = list(ALGORITHMS.keys())
        self.algo_var = tk.StringVar(value=algo_names[0] if algo_names else "")
        self.algo_combo = ttk.Combobox(
            self.sidebar,
            textvariable=self.algo_var,
            values=algo_names,
            state="readonly",
            font=("Segoe UI", 9),
        )
        self.algo_combo.pack(side=tk.TOP, fill=tk.X, ipady=3, pady=(0, 14))
        self.algo_combo.bind("<<ComboboxSelected>>", self._on_algorithm_changed)

        # Картка параметрів алгоритму
        self.params_card = tk.Frame(
            self.sidebar,
            bg="#ffffff",
            padx=12,
            pady=12,
            highlightthickness=1,
            highlightbackground="#e2e8f0",
        )
        self.params_card.pack(side=tk.TOP, fill=tk.X, pady=(0, 16))

        # Кнопка запуску алгоритму
        self.btn_run = tk.Button(
            self.sidebar,
            text="Run Algorithm",
            command=self._on_run_algorithm,
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            disabledforeground="#cbd5e1",
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            pady=8,
            state=tk.DISABLED,
        )
        self.btn_run.pack(side=tk.TOP, fill=tk.X)

        self.status_var = tk.StringVar(
            value="Завантажте зображення або папку для початку роботи."
        )
        self.status_label = tk.Label(
            self.sidebar,
            textvariable=self.status_var,
            bg="#f8fafc",
            fg="#64748b",
            font=("Segoe UI", 8),
            wraplength=255,
            justify=tk.LEFT,
            anchor="w",
        )
        self.status_label.pack(side=tk.TOP, fill=tk.X, pady=(12, 0))

        # --- Права частина: дві зони зображень строго однакового розміру ---
        images_area = tk.Frame(content_frame, bg="#f4f6f9")
        images_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        images_area.columnconfigure(0, weight=1, uniform="equal_image_zone")
        images_area.columnconfigure(1, weight=1, uniform="equal_image_zone")
        images_area.rowconfigure(0, weight=1)

        # Ліва зона — Оригінальне зображення
        orig_panel = tk.Frame(
            images_area,
            bg="#1e222a",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
        )
        orig_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        orig_panel.pack_propagate(False)

        orig_header = tk.Frame(orig_panel, bg="#252b37", padx=10, pady=5)
        orig_header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(
            orig_header,
            text="Original Image",
            bg="#252b37",
            fg="#e2e8f0",
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT)

        self.orig_canvas = tk.Canvas(orig_panel, bg="#161920", highlightthickness=0)
        self.orig_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Права зона — Оброблене зображення
        proc_panel = tk.Frame(
            images_area,
            bg="#1e222a",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
        )
        proc_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        proc_panel.pack_propagate(False)

        proc_header = tk.Frame(proc_panel, bg="#252b37", padx=10, pady=5)
        proc_header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(
            proc_header,
            text="Processed Image",
            bg="#252b37",
            fg="#e2e8f0",
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.LEFT)

        self.proc_canvas = tk.Canvas(proc_panel, bg="#161920", highlightthickness=0)
        self.proc_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.orig_canvas.bind("<Configure>", lambda _e: self._redraw_canvases())
        self.proc_canvas.bind("<Configure>", lambda _e: self._redraw_canvases())

    def _on_algorithm_changed(self, _event: tk.Event | None = None) -> None:
        self._rebuild_param_controls()

    def _rebuild_param_controls(self) -> None:
        """Динамічно перебудовує повзунки та випадаючі списки параметрів під обраний алгоритм."""
        for child in self.params_card.winfo_children():
            child.destroy()
        self.param_getters.clear()

        algo_name = self.algo_var.get()
        specs = ALGORITHM_PARAMS.get(algo_name, [])

        if not specs:
            tk.Label(
                self.params_card,
                text="Немає додаткових параметрів",
                bg="#ffffff",
                fg="#64748b",
                font=("Segoe UI", 9),
            ).pack(anchor="w")
            return

        for idx, spec in enumerate(specs):
            p_name = spec["name"]
            p_label = spec["label"]
            p_type = spec["type"]
            default_val = spec["default"]

            row_frame = tk.Frame(self.params_card, bg="#ffffff")
            row_frame.pack(side=tk.TOP, fill=tk.X, pady=(0 if idx == 0 else 8, 0))

            if p_type == "slider":
                min_val = float(spec["min"])
                max_val = float(spec["max"])
                step_val = float(spec["step"])
                is_int = isinstance(default_val, int) and step_val.is_integer()

                title_var = tk.StringVar(
                    value=f"{p_label}: {int(default_val) if is_int else f'{default_val:.2f}'.rstrip('0').rstrip('.')}"
                )
                tk.Label(
                    row_frame,
                    textvariable=title_var,
                    bg="#ffffff",
                    fg="#334155",
                    font=("Segoe UI", 9, "bold"),
                    anchor="w",
                ).pack(side=tk.TOP, fill=tk.X, pady=(0, 2))

                scale = tk.Scale(
                    row_frame,
                    from_=min_val,
                    to=max_val,
                    resolution=step_val,
                    orient=tk.HORIZONTAL,
                    showvalue=False,
                    bg="#111111",
                    fg="#1e293b",
                    troughcolor="#cbd5e1",
                    activebackground="#2563eb",
                    highlightthickness=0,
                    bd=0,
                    sliderrelief=tk.FLAT,
                    length=220,
                )
                scale.set(default_val)

                def _on_slide(val_str: str, tv=title_var, lbl=p_label, as_int=is_int) -> None:
                    v = float(val_str)
                    formatted = str(int(round(v))) if as_int else f"{v:.2f}".rstrip("0").rstrip(".")
                    tv.set(f"{lbl}: {formatted}")

                scale.configure(command=_on_slide)
                scale.pack(side=tk.TOP, fill=tk.X)

                self.param_getters[p_name] = (
                    lambda sc=scale, as_int=is_int: int(round(sc.get())) if as_int else float(sc.get())
                )

            elif p_type == "combo":
                tk.Label(
                    row_frame,
                    text=f"{p_label}:",
                    bg="#ffffff",
                    fg="#334155",
                    font=("Segoe UI", 9, "bold"),
                    anchor="w",
                ).pack(side=tk.TOP, fill=tk.X, pady=(0, 3))

                combo_var = tk.StringVar(value=str(default_val))
                combo = ttk.Combobox(
                    row_frame,
                    textvariable=combo_var,
                    values=[str(opt) for opt in spec["options"]],
                    state="readonly",
                    font=("Segoe UI", 9),
                )
                combo.pack(side=tk.TOP, fill=tk.X, ipady=2)

                self.param_getters[p_name] = lambda cv=combo_var: cv.get()

    def _collect_current_params(self) -> dict:
        """Збирає поточні значення параметрів із віджетів панелі."""
        return {name: getter() for name, getter in self.param_getters.items()}

    def _update_info_bar(self) -> None:
        """Оновлює єдиний нижній інформаційний бар: Native, Size_orig, Size_proc, Time, Saved."""
        for child in self.info_items_frame.winfo_children():
            child.destroy()

        if not self.loaded_items or not (0 <= self.current_index < len(self.loaded_items)):
            pairs = [
                ("Native", "-"),
                ("Size_orig", "-"),
                ("Size_proc", "-"),
                ("Time", "-"),
                ("Saved", "-"),
            ]
        else:
            orig_img, orig_path = self.loaded_items[self.current_index]
            orig_w, orig_h = orig_img.size

            if self.current_index in self.processed_items:
                proc_info = self.processed_items[self.current_index]
                saved_path: Path | None = proc_info["saved_path"]
                proc_time: float | None = proc_info["time"]
                size_proc_str = _format_file_size(saved_path)
                time_str = _format_proc_time(proc_time)
                saved_str = _format_relative_path(saved_path)
            else:
                size_proc_str = "-"
                time_str = "-"
                saved_str = "-"

            pairs = [
                ("Native", f"{orig_w}×{orig_h} px"),
                ("Size_orig", _format_file_size(orig_path)),
                ("Size_proc", size_proc_str),
                ("Time", time_str),
                ("Saved", saved_str),
            ]

        for idx, (label_text, value_text) in enumerate(pairs):
            tk.Label(
                self.info_items_frame,
                text=f"{label_text}:",
                bg="#1c2331",
                fg="#94a3b8",
                font=("Segoe UI", 9),
            ).pack(side=tk.LEFT, padx=(0 if idx == 0 else 18, 4))

            tk.Label(
                self.info_items_frame,
                text=value_text,
                bg="#1c2331",
                fg="#f8fafc",
                font=("Segoe UI", 9, "bold"),
            ).pack(side=tk.LEFT)

    def _on_load_file(self) -> None:
        if self.is_processing:
            return
        ext_pattern = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        file_path = filedialog.askopenfilename(
            title="Оберіть зображення",
            initialdir=str(Path("./images").resolve()) if Path("./images").is_dir() else ".",
            filetypes=[
                ("Зображення", ext_pattern),
                ("Усі файли", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            img, path = load_image(file_path)
            self.loaded_items = [(img, path)]
            self.processed_items.clear()
            self.current_index = 0
            self._update_navigation_state()
            self._redraw_canvases()
            self._update_info_bar()
            self.status_var.set(f"Завантажено файл: {path.name}")
        except Exception as e:
            messagebox.showerror("Помилка завантаження", f"Не вдалося відкрити файл:\n{e}")

    def _on_load_folder(self) -> None:
        if self.is_processing:
            return
        folder_path = filedialog.askdirectory(
            title="Оберіть папку із зображеннями",
            initialdir=str(Path("./images").resolve()) if Path("./images").is_dir() else ".",
        )
        if not folder_path:
            return

        try:
            items = load_images_from_folder(folder_path)
            if not items:
                messagebox.showwarning(
                    "Порожня папка",
                    "У вибраній папці не знайдено підтримуваних зображень.",
                )
                return

            self.loaded_items = items
            self.processed_items.clear()
            self.current_index = 0
            self._update_navigation_state()
            self._redraw_canvases()
            self._update_info_bar()
            self.status_var.set(
                f"Завантажено папку '{Path(folder_path).name}' ({len(items)} зображень)."
            )
        except Exception as e:
            messagebox.showerror("Помилка завантаження папки", f"Не вдалося прочитати папку:\n{e}")

    def _update_navigation_state(self) -> None:
        total = len(self.loaded_items)
        has_items = total > 0

        self.btn_run.config(state=tk.NORMAL if (has_items and not self.is_processing) else tk.DISABLED)

        if has_items:
            names = [f"[{i + 1}/{total}] {path.name}" for i, (_, path) in enumerate(self.loaded_items)]
            self.file_combo.config(values=names, state="readonly" if not self.is_processing else "disabled")
            self.file_combo.current(self.current_index)
            self.btn_prev.config(
                state=tk.NORMAL if (self.current_index > 0 and not self.is_processing) else tk.DISABLED
            )
            self.btn_next.config(
                state=tk.NORMAL if (self.current_index < total - 1 and not self.is_processing) else tk.DISABLED
            )
        else:
            self.file_combo.config(values=[], state="disabled")
            self.file_var.set("Не обрано")
            self.btn_prev.config(state=tk.DISABLED)
            self.btn_next.config(state=tk.DISABLED)

    def _on_prev_image(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._update_navigation_state()
            self._redraw_canvases()
            self._update_info_bar()

    def _on_next_image(self) -> None:
        if self.current_index < len(self.loaded_items) - 1:
            self.current_index += 1
            self._update_navigation_state()
            self._redraw_canvases()
            self._update_info_bar()

    def _on_select_file_combo(self, _event: tk.Event) -> None:
        idx = self.file_combo.current()
        if 0 <= idx < len(self.loaded_items):
            self.current_index = idx
            self._update_navigation_state()
            self._redraw_canvases()
            self._update_info_bar()

    def _on_run_algorithm(self) -> None:
        if not self.loaded_items or self.is_processing:
            return

        algo_name = self.algo_var.get()
        if algo_name not in ALGORITHMS:
            messagebox.showerror("Помилка", "Оберіть алгоритм зі списку.")
            return

        runner_fn, suffix = ALGORITHMS[algo_name]
        params = self._collect_current_params()

        self.is_processing = True
        self.btn_load_file.config(state=tk.DISABLED)
        self.btn_load_folder.config(state=tk.DISABLED)
        self.algo_combo.config(state="disabled")
        self._update_navigation_state()

        thread = threading.Thread(
            target=self._process_worker,
            args=(algo_name, runner_fn, suffix, params),
            daemon=True,
        )
        thread.start()

    def _process_worker(
        self,
        algo_name: str,
        runner_fn: callable,
        suffix: str,
        params: dict,
    ) -> None:
        total = len(self.loaded_items)
        saved_count = 0

        for idx, (orig_img, orig_path) in enumerate(self.loaded_items):
            self.root.after(
                0,
                self.status_var.set,
                f"Обробка [{idx + 1}/{total}]: {orig_path.name}...",
            )
            try:
                result_img = runner_fn(orig_img, **params)
                last_times = list(timer.last_durations.values())
                elapsed_sec = last_times[-1] if last_times else 0.0

                saved_path = save_image(
                    img=result_img,
                    original_path=orig_path,
                    output_path=self.output_dir,
                    suffix=suffix,
                )
                metrics_dict = compute_image_metrics(orig_img, result_img)
                save_image_metrics_json(
                    original_path=orig_path,
                    saved_image_path=saved_path,
                    algorithm_name=algo_name,
                    parameters=params,
                    processing_time_sec=elapsed_sec,
                    metrics=metrics_dict,
                    output_dir=self.output_dir,
                )
                self.processed_items[idx] = {
                    "img": result_img,
                    "saved_path": saved_path,
                    "time": elapsed_sec,
                    "metrics": metrics_dict,
                }
                if saved_path is not None:
                    saved_count += 1

                self.root.after(0, self._show_processed_index, idx, total)
            except Exception as e:
                self.root.after(
                    0,
                    lambda err=e, name=orig_path.name: messagebox.showerror(
                        "Помилка обробки",
                        f"Помилка під час обробки файлу {name}:\n{err}",
                    ),
                )

        self.root.after(0, self._on_processing_complete, algo_name, saved_count, total)

    def _show_processed_index(self, idx: int, total: int) -> None:
        self.current_index = idx
        if self.loaded_items:
            names = [f"[{i + 1}/{total}] {path.name}" for i, (_, path) in enumerate(self.loaded_items)]
            self.file_combo.config(values=names)
            self.file_combo.current(self.current_index)
        self._redraw_canvases()
        self._update_info_bar()

    def _on_processing_complete(self, algo_name: str, saved_count: int, total: int) -> None:
        self.is_processing = False
        self.btn_load_file.config(state=tk.NORMAL)
        self.btn_load_folder.config(state=tk.NORMAL)
        self.algo_combo.config(state="readonly")
        self._update_navigation_state()
        self._redraw_canvases()
        self._update_info_bar()

        self.status_var.set(
            f"Готово!\nАлгоритм '{algo_name}' збережено {saved_count}/{total} у '{_format_relative_path(self.output_dir)}'."
        )

    def _redraw_canvases(self) -> None:
        self.orig_canvas.delete("all")
        self.proc_canvas.delete("all")

        if not self.loaded_items or not (0 <= self.current_index < len(self.loaded_items)):
            self._draw_placeholder(self.orig_canvas, "Зображення не завантажено")
            self._draw_placeholder(self.proc_canvas, "Очікування результату обробки")
            return

        orig_img, _ = self.loaded_items[self.current_index]
        self._orig_photo = self._render_image_on_canvas(self.orig_canvas, orig_img)

        if self.current_index in self.processed_items:
            proc_img = self.processed_items[self.current_index]["img"]
            self._proc_photo = self._render_image_on_canvas(self.proc_canvas, proc_img)
        else:
            self._draw_placeholder(
                self.proc_canvas,
                "Налаштуйте параметри зліва\nта натисніть «Run Algorithm»",
            )

    @staticmethod
    def _draw_placeholder(canvas: tk.Canvas, text: str) -> None:
        w = max(canvas.winfo_width(), 200)
        h = max(canvas.winfo_height(), 200)
        canvas.create_text(
            w // 2,
            h // 2,
            text=text,
            fill="#64748b",
            font=("Segoe UI", 11),
            justify=tk.CENTER,
        )

    @staticmethod
    def _render_image_on_canvas(canvas: tk.Canvas, img: Image.Image) -> ImageTk.PhotoImage | None:
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        if canvas_w <= 10 or canvas_h <= 10:
            return None

        img_w, img_h = img.size
        if img_w <= 0 or img_h <= 0:
            return None

        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = img.convert("RGB").resize((new_w, new_h), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(resized)
        canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo, anchor=tk.CENTER)
        return photo


def launch_gui(output_dir: str | Path = r".\output") -> None:
    root = tk.Tk()
    SpaceAnalyzeApp(root, output_dir=output_dir)
    root.mainloop()
