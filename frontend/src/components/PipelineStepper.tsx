import { Activity, Archive, AudioLines, Bot, FileAudio, FolderOpen, GitBranch, Wrench } from 'lucide-react'
import { pipelineStages, stageProgressIndex, usePipeline } from '../state/PipelineContext'

const icons = [FolderOpen, FileAudio, AudioLines, Bot, Activity, GitBranch, Wrench, Archive]

export function PipelineStepper() {
  const { job } = usePipeline()
  const progressIndex = stageProgressIndex(job?.progress, job?.status)
  return (
    <section className="glassPanel pipelinePanel" data-reveal aria-labelledby="pipeline-stepper-title">
      <div className="panelHeader">
        <div><p className="sectionKicker">PIPELINE</p><h2 id="pipeline-stepper-title">处理流水线</h2></div>
        <span className="panelMeta">{progressIndex >= 0 ? `已完成 ${Math.min(progressIndex + 1, pipelineStages.length)} / ${pipelineStages.length}` : '等待运行'}</span>
      </div>
      <div className="pipelineSteps" role="list" aria-label="音频处理流水线">
        {pipelineStages.map((stage, index) => {
          const Icon = icons[index]
          const complete = progressIndex > index || job?.status === 'succeeded'
          const active = progressIndex === index && job?.status !== 'succeeded'
          return (
            <div className="pipelineStepWrap" key={stage.id}>
              <div className={`pipelineStep ${complete ? 'completed' : ''} ${active ? 'current' : ''}`} role="listitem" title={stage.description}>
                <span className="stepIcon"><Icon size={16} /></span>
                <span className="stepCopy"><strong>{stage.label}</strong><small>{complete ? '已完成' : active ? '处理中' : '待执行'}</small></span>
              </div>
              {index < pipelineStages.length - 1 && <span className={`pipelineConnector ${complete ? 'completed' : ''} ${active ? 'current' : ''}`} aria-hidden="true"><svg viewBox="0 0 32 4" preserveAspectRatio="none"><line className="pipelineLine" x1="1" y1="2" x2="31" y2="2" /></svg></span>}
            </div>
          )
        })}
      </div>
    </section>
  )
}
