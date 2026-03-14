"""Tests for scripts/random_rollout.py — covering AC-6 and AC-7 from hello-crafter-spec.md."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Path to the script under test
SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "random_rollout.py"

# Valid action names from the spec
VALID_ACTION_NAMES = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "place_table",
    "make_wood_pickaxe",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_rollout(tmp_path: Path, extra_args: list[str] | None = None, max_steps: int = 10) -> Path:
    """Run the random_rollout.py script and return the output directory."""
    output_dir = tmp_path / "rollout_output"
    cmd = [
        "uv",
        "run",
        "python",
        str(SCRIPT_PATH),
        "--seed",
        "42",
        "--output-dir",
        str(output_dir),
        "--max-steps",
        str(max_steps),
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_PATH.parent.parent)
    assert result.returncode == 0, (
        f"random_rollout.py exited with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return output_dir


# ---------------------------------------------------------------------------
# Unit-style tests: episode metadata JSON structure (no Crafter needed)
# ---------------------------------------------------------------------------


class TestEpisodeJsonStructure:
    """Verify the expected JSON schema for episode.json (structure only, no I/O)."""

    REQUIRED_KEYS = {
        "seed",
        "num_steps",
        "total_reward",
        "actions_taken",
        "action_names_taken",
        "final_inventory",
        "achievements",
    }

    def test_all_required_keys_defined(self):
        """All required episode.json keys from the spec must be present in our expected set."""
        assert self.REQUIRED_KEYS == {
            "seed",
            "num_steps",
            "total_reward",
            "actions_taken",
            "action_names_taken",
            "final_inventory",
            "achievements",
        }

    def test_script_file_exists(self):
        """scripts/random_rollout.py must exist at the expected path."""
        assert SCRIPT_PATH.exists(), f"Expected script at {SCRIPT_PATH} but it does not exist"


# ---------------------------------------------------------------------------
# AC-6: Random rollout script produces correct outputs
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRandomRolloutOutputs:
    """AC-6: Integration tests for the random rollout script outputs."""

    def test_output_directory_is_created(self, tmp_path):
        """Running the script must create the output directory."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        assert output_dir.is_dir(), f"Output directory {output_dir} was not created"

    def test_frames_directory_is_created(self, tmp_path):
        """Running the script must create a frames/ subdirectory."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        frames_dir = output_dir / "frames"
        assert frames_dir.is_dir(), f"frames/ directory not found in {output_dir}"

    def test_at_least_one_frame_saved(self, tmp_path):
        """frames/ must contain at least 1 PNG file."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        frames_dir = output_dir / "frames"
        png_files = sorted(frames_dir.glob("*.png"))
        assert len(png_files) >= 1, f"Expected at least 1 PNG in frames/, found {len(png_files)}"

    def test_frame_files_named_correctly(self, tmp_path):
        """Frame files must follow the naming pattern frame_NNN.png (zero-padded 3 digits)."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        frames_dir = output_dir / "frames"
        png_files = sorted(frames_dir.glob("*.png"))
        for i, f in enumerate(png_files):
            expected_name = f"frame_{i:03d}.png"
            assert f.name == expected_name, f"Expected frame file '{expected_name}', got '{f.name}'"

    def test_frames_are_valid_images(self, tmp_path):
        """Each PNG in frames/ must be loadable by imageio with shape (64, 64, 3)."""
        import imageio.v3 as iio

        output_dir = run_rollout(tmp_path, max_steps=5)
        frames_dir = output_dir / "frames"
        png_files = sorted(frames_dir.glob("*.png"))
        assert len(png_files) >= 1, "No frames to check"
        for png_file in png_files:
            img = iio.imread(str(png_file))
            assert img.shape == (64, 64, 3), (
                f"Expected image shape (64, 64, 3) for {png_file.name}, got {img.shape}"
            )

    def test_episode_json_exists(self, tmp_path):
        """episode.json must exist in the output directory."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        episode_json = output_dir / "episode.json"
        assert episode_json.is_file(), f"episode.json not found in {output_dir}"

    def test_episode_json_is_valid_json(self, tmp_path):
        """episode.json must be parseable JSON."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        episode_json = output_dir / "episode.json"
        with open(episode_json) as f:
            data = json.load(f)  # raises if invalid JSON
        assert isinstance(data, dict)

    def test_episode_json_has_all_required_keys(self, tmp_path):
        """episode.json must contain all keys from the spec."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        episode_json = output_dir / "episode.json"
        with open(episode_json) as f:
            data = json.load(f)
        required_keys = {
            "seed",
            "num_steps",
            "total_reward",
            "actions_taken",
            "action_names_taken",
            "final_inventory",
            "achievements",
        }
        missing = required_keys - set(data.keys())
        assert not missing, f"episode.json is missing keys: {missing}"

    def test_episode_json_seed_matches_arg(self, tmp_path):
        """episode.json['seed'] must equal the --seed argument (42)."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        with open(output_dir / "episode.json") as f:
            data = json.load(f)
        assert data["seed"] == 42, f"Expected seed=42 in episode.json, got {data['seed']}"

    def test_episode_json_num_steps_equals_frame_count(self, tmp_path):
        """episode.json['num_steps'] must equal the number of PNG frames saved."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        frames_dir = output_dir / "frames"
        num_frames = len(list(frames_dir.glob("*.png")))
        with open(output_dir / "episode.json") as f:
            data = json.load(f)
        assert data["num_steps"] == num_frames, (
            f"num_steps ({data['num_steps']}) != number of frames ({num_frames})"
        )

    def test_episode_json_actions_taken_length(self, tmp_path):
        """episode.json['actions_taken'] must have length equal to num_steps."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        with open(output_dir / "episode.json") as f:
            data = json.load(f)
        assert len(data["actions_taken"]) == data["num_steps"], (
            f"actions_taken length ({len(data['actions_taken'])}) != num_steps ({data['num_steps']})"
        )

    def test_episode_json_action_names_taken_length(self, tmp_path):
        """episode.json['action_names_taken'] must have length equal to num_steps."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        with open(output_dir / "episode.json") as f:
            data = json.load(f)
        assert len(data["action_names_taken"]) == data["num_steps"], (
            f"action_names_taken length ({len(data['action_names_taken'])}) "
            f"!= num_steps ({data['num_steps']})"
        )

    def test_episode_json_action_names_taken_valid(self, tmp_path):
        """Every entry in action_names_taken must be in the valid action names list."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        with open(output_dir / "episode.json") as f:
            data = json.load(f)
        for i, name in enumerate(data["action_names_taken"]):
            assert name in VALID_ACTION_NAMES, (
                f"action_names_taken[{i}] = '{name}' is not a valid action name. "
                f"Valid: {VALID_ACTION_NAMES}"
            )

    def test_episode_json_actions_taken_valid_indices(self, tmp_path):
        """Every entry in actions_taken must be an integer in range 0-7."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        with open(output_dir / "episode.json") as f:
            data = json.load(f)
        for i, action in enumerate(data["actions_taken"]):
            assert isinstance(action, int), f"actions_taken[{i}] must be an int, got {type(action)}"
            assert 0 <= action <= 7, f"actions_taken[{i}] = {action} is out of valid range 0-7"

    def test_episode_json_total_reward_is_numeric(self, tmp_path):
        """episode.json['total_reward'] must be a number (int or float)."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        with open(output_dir / "episode.json") as f:
            data = json.load(f)
        assert isinstance(data["total_reward"], (int, float)), (
            f"total_reward must be numeric, got {type(data['total_reward'])}"
        )

    def test_episode_json_final_inventory_is_dict(self, tmp_path):
        """episode.json['final_inventory'] must be a dict."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        with open(output_dir / "episode.json") as f:
            data = json.load(f)
        assert isinstance(data["final_inventory"], dict), (
            f"final_inventory must be a dict, got {type(data['final_inventory'])}"
        )

    def test_episode_json_achievements_is_dict(self, tmp_path):
        """episode.json['achievements'] must be a dict."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        with open(output_dir / "episode.json") as f:
            data = json.load(f)
        assert isinstance(data["achievements"], dict), (
            f"achievements must be a dict, got {type(data['achievements'])}"
        )

    def test_max_steps_respected(self, tmp_path):
        """The script must not exceed --max-steps steps."""
        output_dir = run_rollout(tmp_path, max_steps=7)
        with open(output_dir / "episode.json") as f:
            data = json.load(f)
        assert data["num_steps"] <= 7, f"num_steps ({data['num_steps']}) exceeded --max-steps 7"

    def test_action_names_and_indices_consistent(self, tmp_path):
        """actions_taken[i] must correspond to action_names_taken[i] via the valid mapping."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        with open(output_dir / "episode.json") as f:
            data = json.load(f)
        for i, (idx, name) in enumerate(zip(data["actions_taken"], data["action_names_taken"])):
            expected_name = VALID_ACTION_NAMES[idx]
            assert name == expected_name, (
                f"Step {i}: actions_taken={idx} but action_names_taken='{name}'; "
                f"expected '{expected_name}'"
            )


# ---------------------------------------------------------------------------
# AC-7: Video output (--save-video flag)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestVideoOutput:
    """AC-7: --save-video flag produces a valid rollout.mp4."""

    def test_no_video_without_flag(self, tmp_path):
        """Without --save-video, rollout.mp4 must NOT be created."""
        output_dir = run_rollout(tmp_path, max_steps=5)
        video_file = output_dir / "rollout.mp4"
        assert not video_file.exists(), (
            "rollout.mp4 should not exist when --save-video is not passed"
        )

    def test_video_created_with_flag(self, tmp_path):
        """With --save-video, rollout.mp4 must be created in the output directory."""
        output_dir = run_rollout(tmp_path, extra_args=["--save-video"], max_steps=5)
        video_file = output_dir / "rollout.mp4"
        assert video_file.is_file(), (
            f"rollout.mp4 not found in {output_dir} after running with --save-video"
        )

    def test_video_is_non_empty(self, tmp_path):
        """rollout.mp4 must be a non-empty file (> 0 bytes)."""
        output_dir = run_rollout(tmp_path, extra_args=["--save-video"], max_steps=5)
        video_file = output_dir / "rollout.mp4"
        assert video_file.stat().st_size > 0, "rollout.mp4 is empty"

    def test_video_is_readable_by_imageio(self, tmp_path):
        """rollout.mp4 must be readable as a video by imageio."""
        import imageio.v3 as iio

        output_dir = run_rollout(tmp_path, extra_args=["--save-video"], max_steps=5)
        video_file = output_dir / "rollout.mp4"
        # Read first frame to confirm it's a valid video
        frames = list(iio.imiter(str(video_file), plugin="pyav"))
        assert len(frames) >= 1, "rollout.mp4 contains no readable frames"


# ---------------------------------------------------------------------------
# AC-6: Console output format (checked via captured stdout)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestConsoleOutput:
    """AC-6 (extended): Script produces expected console output format."""

    def _run_and_capture(self, tmp_path: Path, max_steps: int = 5) -> tuple[str, Path]:
        output_dir = tmp_path / "rollout_console_test"
        cmd = [
            "uv",
            "run",
            "python",
            str(SCRIPT_PATH),
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
            "--max-steps",
            str(max_steps),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_PATH.parent.parent)
        assert result.returncode == 0, (
            f"Script failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        return result.stdout, output_dir

    def test_episode_done_line_in_output(self, tmp_path):
        """stdout must contain 'Episode done:' summary line."""
        stdout, _ = self._run_and_capture(tmp_path)
        assert "Episode done:" in stdout, f"'Episode done:' not found in stdout:\n{stdout}"

    def test_saved_frames_line_in_output(self, tmp_path):
        """stdout must contain a line about saved frames."""
        stdout, _ = self._run_and_capture(tmp_path)
        assert "frame" in stdout.lower(), f"No mention of saved frames in stdout:\n{stdout}"

    def test_saved_metadata_line_in_output(self, tmp_path):
        """stdout must contain a line about saved episode metadata."""
        stdout, _ = self._run_and_capture(tmp_path)
        assert "episode" in stdout.lower(), f"No mention of episode metadata in stdout:\n{stdout}"

    def test_step_log_lines_in_output(self, tmp_path):
        """stdout must contain per-step log lines with 'action:' and 'reward:'."""
        stdout, _ = self._run_and_capture(tmp_path, max_steps=5)
        assert "action:" in stdout, f"No 'action:' found in stdout:\n{stdout}"
        assert "reward:" in stdout, f"No 'reward:' found in stdout:\n{stdout}"
