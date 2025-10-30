"""
Task 4: Analysis and Comparison
================================

This module implements:
- Model comparison (DL vs RL)
- Decision disagreement analysis
- Metric explanations
- Future recommendations
"""

import numpy as np
import pandas as pd
import json


class ModelComparison:
    """Compare Deep Learning and RL models"""
    
    def __init__(self, dl_results, rl_results, X_test, y_test):
        self.dl_results = dl_results
        self.rl_results = rl_results
        self.X_test = X_test
        self.y_test = y_test
    
    def compare_decisions(self, threshold=0.5):
        """
        Compare where models make different decisions
        Identify cases where DL and RL disagree
        """
        print("\n" + "="*80)
        print("MODEL COMPARISON: DECISION ANALYSIS")
        print("="*80)
        
        # DL policy: approve if prob(default) < threshold
        dl_approve = (self.dl_results['y_pred_proba'] < threshold).astype(int)
        
        # RL policy
        rl_approve = self.rl_results['actions']
        
        # Agreement analysis
        agreement = (dl_approve == rl_approve).mean()
        
        print(f"\n📊 Overall Agreement: {agreement:.2%}")
        print(f"DL Approval Rate:     {dl_approve.mean():.2%}")
        print(f"RL Approval Rate:     {rl_approve.mean():.2%}")
        
        # Find cases where models disagree
        disagree_idx = np.where(dl_approve != rl_approve)[0]
        
        print(f"\n🔍 Cases of disagreement: {len(disagree_idx)} ({len(disagree_idx)/len(dl_approve)*100:.1f}%)")
        
        # High-risk applicants that RL approves but DL denies
        high_risk_rl_approves = disagree_idx[
            (dl_approve[disagree_idx] == 0) & (rl_approve[disagree_idx] == 1)
        ]
        
        print(f"\n⚠️  High-risk loans approved by RL but denied by DL: {len(high_risk_rl_approves)}")
        
        if len(high_risk_rl_approves) > 0:
            print("\n" + "-"*80)
            print("EXAMPLE CASES (Why RL approves high-risk loans):")
            print("-"*80)
            
            for i, idx in enumerate(high_risk_rl_approves[:5]):
                print(f"\n📋 Case {i+1} (Index: {idx}):")
                print(f"  DL Default Probability:  {self.dl_results['y_pred_proba'][idx]:.3f} (HIGH RISK)")
                print(f"  DL Decision:             DENY (prob > {threshold})")
                print(f"  RL Q(deny):              {self.rl_results['q_deny'][idx]:.2f}")
                print(f"  RL Q(approve):           {self.rl_results['q_approve'][idx]:.2f}")
                print(f"  RL Decision:             APPROVE (Q(approve) > Q(deny))")
                
                actual_outcome = "DEFAULT" if self.y_test.iloc[idx] == 1 else "FULLY PAID"
                print(f"  Actual Outcome:          {actual_outcome}")
                
                # Explain why RL approved
                if self.rl_results['q_approve'][idx] > self.rl_results['q_deny'][idx]:
                    print(f"\n  💡 Why RL Approved:")
                    print(f"     RL optimizes for EXPECTED PROFIT, not just accuracy.")
                    print(f"     Even with some default risk, if expected value is positive,")
                    print(f"     RL will approve (e.g., high interest rate compensates risk).")
        
        # Low-risk but low-profit loans that DL approves but RL denies
        low_risk_rl_denies = disagree_idx[
            (dl_approve[disagree_idx] == 1) & (rl_approve[disagree_idx] == 0)
        ]
        
        print(f"\n\n💰 Low-profit loans approved by DL but denied by RL: {len(low_risk_rl_denies)}")
        
        if len(low_risk_rl_denies) > 0:
            print("\n" + "-"*80)
            print("EXAMPLE CASES (Why RL denies low-risk loans):")
            print("-"*80)
            
            for i, idx in enumerate(low_risk_rl_denies[:3]):
                print(f"\n📋 Case {i+1} (Index: {idx}):")
                print(f"  DL Default Probability:  {self.dl_results['y_pred_proba'][idx]:.3f} (LOW RISK)")
                print(f"  DL Decision:             APPROVE (prob < {threshold})")
                print(f"  RL Q(deny):              {self.rl_results['q_deny'][idx]:.2f}")
                print(f"  RL Q(approve):           {self.rl_results['q_approve'][idx]:.2f}")
                print(f"  RL Decision:             DENY (Q(deny) > Q(approve))")
                
                print(f"\n  💡 Why RL Denied:")
                print(f"     RL focuses on PROFITABILITY, not just safety.")
                print(f"     Low-risk but low-interest loans may have negative")
                print(f"     expected value after considering opportunity cost.")
        
        print("\n" + "="*80)
        
        return {
            'agreement': agreement,
            'dl_approval_rate': dl_approve.mean(),
            'rl_approval_rate': rl_approve.mean(),
            'disagreement_cases': len(disagree_idx),
            'high_risk_rl_approves': len(high_risk_rl_approves),
            'low_risk_rl_denies': len(low_risk_rl_denies)
        }
    
    def explain_metrics(self):
        """Explain the difference in metrics between DL and RL"""
        print("\n" + "="*80)
        print("METRIC EXPLANATION: DL vs RL")
        print("="*80)
        
        print("\n📊 DEEP LEARNING MODEL METRICS")
        print("-" * 80)
        print("1. AUC-ROC (Area Under ROC Curve):")
        print("   • Measures: Ability to discriminate between classes")
        print("   • Range: 0 to 1 (0.5 = random, 1.0 = perfect)")
        print("   • Tells us: How well can the model RANK risky vs safe applicants?")
        print("   • Business value: Identifies who is likely to default")
        print("   • Focus: PREDICTION ACCURACY")
        
        print("\n2. F1-Score:")
        print("   • Measures: Harmonic mean of precision and recall")
        print("   • Range: 0 to 1 (1.0 = perfect)")
        print("   • Tells us: How accurate is the CLASSIFICATION overall?")
        print("   • Business value: Balances false positives and false negatives")
        print("   • Focus: BALANCED CLASSIFICATION ACCURACY")
        
        print("\n\n💰 REINFORCEMENT LEARNING AGENT METRICS")
        print("-" * 80)
        print("1. Estimated Policy Value:")
        print("   • Measures: Expected cumulative reward (profit) per loan")
        print("   • Units: Dollars ($)")
        print("   • Tells us: What is the EXPECTED FINANCIAL RETURN?")
        print("   • Business value: Directly optimizes for profitability")
        print("   • Focus: REWARD MAXIMIZATION")
        
        print("\n2. Total Expected Value:")
        print("   • Measures: Aggregate profit across all decisions")
        print("   • Units: Dollars ($)")
        print("   • Tells us: Total expected profit from the policy")
        print("   • Business value: Overall financial impact")
        
        print("\n\n🎯 KEY DIFFERENCE")
        print("="*80)
        print("DL Model:")
        print("  Objective:     Optimize for PREDICTION ACCURACY")
        print("  Goal:          Correctly identify who will default")
        print("  Behavior:      Tends to be RISK-AVERSE")
        print("  Decision:      Deny if default probability > threshold")
        print("  Best for:      Risk management and compliance")
        
        print("\nRL Agent:")
        print("  Objective:     Optimize for REWARD MAXIMIZATION (profit)")
        print("  Goal:          Maximize expected financial return")
        print("  Behavior:      RISK-AWARE but profit-seeking")
        print("  Decision:      Approve if expected value is positive")
        print("  Best for:      Profitability optimization")
        
        print("\n\n💡 PRACTICAL EXAMPLE")
        print("-" * 80)
        print("High-Risk, High-Interest Loan:")
        print("  • Loan Amount: $30,000")
        print("  • Interest Rate: 18%")
        print("  • Default Probability: 30% (DL prediction)")
        print()
        print("  DL Decision:")
        print("    → DENY (default probability 30% > threshold 20%)")
        print("    → Reasoning: Too risky based on classification")
        print()
        print("  RL Decision:")
        print("    → Calculate expected value:")
        print("      E[Reward] = P(paid) × interest - P(default) × principal")
        print("      E[Reward] = 0.70 × ($5,400) - 0.30 × ($30,000)")
        print("      E[Reward] = $3,780 - $9,000 = -$5,220")
        print("    → DENY (negative expected value)")
        print("    → Reasoning: Not profitable despite high interest")
        print()
        print("  But if interest rate were 25%:")
        print("      E[Reward] = 0.70 × ($7,500) - 0.30 × ($30,000)")
        print("      E[Reward] = $5,250 - $9,000 = -$3,750")
        print("    → Still DENY (still negative)")
        print()
        print("  The key: RL approves ONLY when expected profit is positive,")
        print("           considering both risk AND reward!")
        print("="*80)
    
    def future_recommendations(self):
        """Provide recommendations for future work"""
        print("\n" + "="*80)
        print("LIMITATIONS AND FUTURE WORK")
        print("="*80)
        
        print("\n🚧 CURRENT LIMITATIONS")
        print("-" * 80)
        
        print("\n1. Offline RL Assumptions:")
        print("   • Assumes historical approval policy was reasonable")
        print("   • Cannot explore better actions for denied loans (no data)")
        print("   • Potential distributional shift between training and deployment")
        print("   • Limited to observed state-action pairs in dataset")
        
        print("\n2. Simplified Reward Function:")
        print("   • Doesn't account for partial repayments")
        print("   • Ignores time value of money (discounting)")
        print("   • No consideration of collection costs")
        print("   • Binary outcome (paid vs default) oversimplifies reality")
        
        print("\n3. Missing Temporal Dynamics:")
        print("   • Doesn't model how borrower behavior changes over time")
        print("   • No sequential decision making (one-shot approval)")
        print("   • Cannot adjust loan terms or monitor during repayment")
        
        print("\n4. Feature Limitations:")
        print("   • Limited to features in historical data")
        print("   • May miss important predictive signals")
        print("   • No external economic indicators")
        print("   • Lack of alternative data sources")
        
        print("\n\n📈 RECOMMENDED NEXT STEPS")
        print("-" * 80)
        
        print("\n1. Data Collection:")
        print("   • Gather more detailed payment history")
        print("   • Collect macroeconomic indicators (GDP, unemployment)")
        print("   • Track employment stability metrics")
        print("   • Add social/behavioral data (with consent)")
        print("   • Include denied loan applications for counterfactual learning")
        
        print("\n2. Model Improvements:")
        print("   • Explore advanced offline RL:")
        print("     - Conservative Q-Learning (CQL)")
        print("     - Implicit Q-Learning (IQL)")
        print("     - Decision Transformer")
        print("   • Implement contextual bandits for online learning")
        print("   • Add uncertainty quantification (Bayesian approaches)")
        print("   • Ensemble methods combining DL and RL")
        print("   • Multi-objective optimization (profit + fairness)")
        
        print("\n3. Business Enhancements:")
        print("   • Implement fairness constraints (equal opportunity)")
        print("   • Add explainability for regulatory compliance (LIME, SHAP)")
        print("   • A/B testing framework for safe deployment")
        print("   • Risk-adjusted pricing based on predictions")
        print("   • Dynamic loan term adjustment")
        
        print("\n4. Production Considerations:")
        print("   • Model monitoring for drift detection")
        print("   • Human-in-the-loop for edge cases")
        print("   • Gradual rollout with safety guardrails")
        print("   • Regular retraining pipeline")
        print("   • Real-time inference optimization")
        
        print("\n\n💡 DEPLOYMENT RECOMMENDATION")
        print("="*80)
        print("\n✅ START WITH DL MODEL FOR:")
        print("  • Better interpretability and regulatory acceptance")
        print("  • Lower risk of unexpected behavior")
        print("  • Proven track record in similar applications")
        print("  • Easier to explain to stakeholders")
        
        print("\n✅ GRADUALLY INTEGRATE RL FOR:")
        print("  • Optimizing approval thresholds")
        print("  • Dynamic pricing strategies")
        print("  • Learning from new data via online RL")
        print("  • Profit optimization in low-risk segments")
        
        print("\n✅ HYBRID APPROACH (RECOMMENDED):")
        print("  1. Use DL for RISK ASSESSMENT")
        print("     → Predicts default probability")
        print("  2. Use RL for DECISION MAKING")
        print("     → Optimizes approval with DL risk as input")
        print("  3. Add HUMAN OVERSIGHT for:")
        print("     → High-value loans (>$50K)")
        print("     → Edge cases (unusual features)")
        print("     → Regulatory compliance review")
        
        print("\n✅ SUCCESS METRICS:")
        print("  • Short-term: Monitor approval rate and default rate")
        print("  • Medium-term: Track actual profit vs predicted")
        print("  • Long-term: Customer lifetime value and satisfaction")
        print("  • Always: Fairness metrics across protected groups")
        
        print("="*80)
    
    def generate_report(self):
        """Generate comprehensive comparison report"""
        print("\n\n" + "="*80)
        print("COMPREHENSIVE COMPARISON REPORT")
        print("="*80)
        
        # Compare decisions
        comparison_stats = self.compare_decisions(threshold=0.5)
        
        # Explain metrics
        self.explain_metrics()
        
        # Future recommendations
        self.future_recommendations()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
        return comparison_stats


if __name__ == "__main__":
    """Example usage"""
    import pickle
    
    print("\n" + "="*80)
    print("TASK 4: ANALYSIS AND COMPARISON")
    print("="*80)
    
    # Load results
    print("\nLoading model results...")
    
    with open('dl_results.json', 'r') as f:
        dl_metrics = json.load(f)
    
    with open('rl_results.json', 'r') as f:
        rl_metrics = json.load(f)
    
    # Load predictions (need to load from numpy/pickle if saved)
    # For this example, we'll show the structure
    print("Note: Full comparison requires loading prediction arrays")
    print("\nDL Model Performance:")
    print(f"  AUC-ROC: {dl_metrics['auc']:.4f}")
    print(f"  F1-Score: {dl_metrics['f1']:.4f}")
    
    print("\nRL Agent Performance:")
    print(f"  Policy Value: ${rl_metrics['policy_value']:,.2f}")
    print(f"  Approval Rate: {rl_metrics['approval_rate']:.2%}")
    
    # For actual comparison, load full results with predictions
    # comparison = ModelComparison(dl_results_full, rl_results_full, X_test, y_test)
    # comparison.generate_report()
    
    print("\n" + "="*80)
    print("TASK 4 COMPLETE: Analysis and Comparison")
    print("="*80)
    print("\nTo run full comparison:")
    print("  1. Train both models (task2 and task3)")
    print("  2. Save prediction arrays")
    print("  3. Load and compare using ModelComparison class")
