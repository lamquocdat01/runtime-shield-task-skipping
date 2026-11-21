-- GateShield.hs -- topic 06, P2 Viec 3 (item 7): Copilot 4.8.1 specification of
--   (1) the LAZY SHIELD L of Theorem 4 (theory/THEORY_06.md sec. 3.4): e_t = valid_t and
--       (pi_t or "the last R-1 output frames were CLOSED"); a missing frame is never OPEN
--       (spec/FSM_06.md sec. 1.4, point 4). Used open loop (pi = any policy trace) and closed
--       loop (the policy reads e) by code/shield_bridge.py.
--   (2) the reduced Deep3 gate controller G1 of spec/FSM_06.md sec. 1.3 -- absolute refresh
--       (pos % T == 0, pos > 0), reuse cap with both comparisons, fsl_0 = W + 1, missing
--       frames tick fsl and are never OPEN -- in three variants computed side by side:
--         o_ge   fixed '>=' ;   o_gt   deployed '>' (the shipped bug) ;
--         o_gtL  deployed '>' with the lazy shield INSIDE the loop (fsl reset by its OPENs).
--       Proposition 4.4 (T >= W + 1, R = W + 1): o_gtL == o_ge for every input sequence;
--       stated below as copilot-theorem properties and proved with Z3 (What4 backend).
-- Parameters R, W, T are compile-time constants (Deep3: T = 15, W = 4, R = 5).
-- Inputs per frame (C globals, set by the caller before step()):
--   shield: pi_in (policy wants OPEN), valid_in (frame present), rst_in (first frame of a video)
--   ctl:    g_in (gate request = hysteresis OR watchdog), valid_in, rst_in
-- Written from the spec; no Deep3/ASMAG code. Generates gate_shield.{c,h}, gate_ctl.{c,h}.
{-# LANGUAGE RebindableSyntax #-}
module Main where

import Control.Monad (void)
import qualified Prelude as P
import System.Environment (getArgs)

import Language.Copilot
import Copilot.Compile.C99 (compile, mkDefaultCSettings)
import Copilot.Theorem.What4 (Solver (..), prove)
import Copilot.Verifier (Verbosity (..), VerifierOptions (..), defaultVerifierOptions,
                         verifyWithOptions)

rB, wB, tB :: Word32
rB = 5     -- refresh bound R of the certificate (I_R)
wB = 4     -- W_maxskip
tB = 15    -- T_refresh

satInc :: Stream Word32 -> Word32 -> Stream Word32
satInc x hi = if x >= constant hi then constant hi else x + 1

-- ------------------------------------------------------------------ (1) lazy shield L
shieldSpec :: Spec
shieldSpec = do
  let piIn    = extern "pi_in" Nothing :: Stream Bool
      validIn = extern "valid_in" Nothing :: Stream Bool
      rstIn   = extern "rst_in" Nothing :: Stream Bool
      -- consecutive CLOSED output frames before the current frame (saturated at R);
      -- a reset starts "with no recent OPEN" (run = R), so the first frame is forced
      runStore = [rB] ++ runNext
      runCur   = if rstIn then constant rB else runStore
      force    = runCur >= constant (rB - 1)
      out      = validIn && (piIn || force)
      runNext  = if out then 0 else satInc runCur rB
  -- the shield's guarantee: a valid frame never ends a run of R CLOSED output frames
  void $ prop "shield_IR" (forAll (not validIn || runNext < constant rB))
  void $ prop "shield_only_add" (forAll (not (piIn && validIn) || out))
  void $ prop "shield_lazy" (forAll (not out || (piIn && validIn) || force))
  trigger "shield_out" true [arg out, arg force, arg runCur]

-- ------------------------------------------------------- (2) G1 controller, 3 variants
ctlSpec :: Spec
ctlSpec = do
  let gIn     = extern "g_in" Nothing :: Stream Bool
      validIn = extern "valid_in" Nothing :: Stream Bool
      rstIn   = extern "rst_in" Nothing :: Stream Bool
      -- position modulo T; refresh at pos % T == 0, pos > 0 (absolute, phase-locked)
      phStore = [0] ++ phCur
      phCur   = if rstIn then 0 else (if (phStore + 1) >= constant tB then 0 else phStore + 1)
      rf      = not rstIn && (phCur == 0)
      fsl0    = constant (wB + 1)
      -- fixed '>='
      fGeS = [wB + 1] ++ fGeN
      fGe  = if rstIn then fsl0 else fGeS
      oGe  = validIn && (gIn || rf || fGe >= constant wB)
      fGeN = if oGe then 0 else satInc fGe (wB + 1)
      -- deployed '>'
      fGtS = [wB + 1] ++ fGtN
      fGt  = if rstIn then fsl0 else fGtS
      oGt  = validIn && (gIn || rf || fGt > constant wB)
      fGtN = if oGt then 0 else satInc fGt (wB + 1)
      -- deployed '>' + lazy shield in the loop (the shield's run counter = closed outputs)
      fLS  = [wB + 1] ++ fLN
      fL   = if rstIn then fsl0 else fLS
      rLS  = [rB] ++ rLN
      rL   = if rstIn then constant rB else rLS
      oGtL = validIn && (gIn || rf || fL > constant wB || rL >= constant (rB - 1))
      fLN  = if oGtL then 0 else satInc fL (wB + 1)
      rLN  = if oGtL then 0 else satInc rL rB
  -- Prop. 4.4 (needs R = W + 1 <= T). The What4 backend proves each property by induction
  -- on itself; "oGtL == oGe" alone is not 1-inductive (first run, 17-10: Invalid), so the
  -- output equality is proved together with the state invariant it follows from.
  void $ prop "prop44_inv" (forAll (fL == fGe && rL == fGe))
  void $ prop "prop44" (forAll (fL == fGe && rL == fGe && oGtL == oGe))
  void $ prop "prop44_output_only" (forAll (oGtL == oGe))
  trigger "ctl_out" true [arg oGe, arg oGt, arg oGtL, arg fGe, arg fGt]

main :: IO ()
main = do
  args <- getArgs
  s1 <- reify shieldSpec
  s2 <- reify ctlSpec
  case args of
    ["compile"] -> do
      compile "gate_shield" s1
      compile "gate_ctl" s2
    ["prove"] -> do
      r1 <- prove Z3 s1
      print r1
      r2 <- prove Z3 s2
      print r2
    _ -> do
      let opts = defaultVerifierOptions { verbosity = Default }
      verifyWithOptions opts mkDefaultCSettings [] "gate_shield" s1
      verifyWithOptions opts mkDefaultCSettings [] "gate_ctl" s2
