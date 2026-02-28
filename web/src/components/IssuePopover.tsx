import React, { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import { ReviewIssue } from '../types'

interface IssuePopoverProps {
    issue: ReviewIssue
    position: { top: number; left: number } | null
    onClose: () => void
}

export function IssuePopoverComponent({ issue, position, onClose }: IssuePopoverProps) {
    const ref = useRef<HTMLDivElement>(null)

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (ref.current && !ref.current.contains(event.target as Node)) {
                onClose()
            }
        }
        document.addEventListener('mousedown', handleClickOutside)
        return () => {
            document.removeEventListener('mousedown', handleClickOutside)
        }
    }, [onClose])

    if (!position) return null

    // Determine header text based on category
    let headerText = 'Informational'
    if (issue.category === 'bug') {
        headerText = 'Bug'
    } else if (issue.category === 'investigation' || issue.severity === 'high' || issue.severity === 'critical') {
        headerText = 'Investigate'
    }

    // Format line range
    const lineRange = issue.citations[0] ? `R${issue.citations[0].startLine}${issue.citations[0].endLine !== issue.citations[0].startLine ? `-${issue.citations[0].endLine}` : ''}` : ''

    // Dynamic positioning
    const [adjustedPosition, setAdjustedPosition] = React.useState(position)

    useEffect(() => {
        if (!ref.current || !position) return

        const rect = ref.current.getBoundingClientRect()
        const viewportHeight = window.innerHeight
        const viewportWidth = window.innerWidth

        let newTop = position.top
        let newLeft = position.left

        // Flip vertically if too close to bottom
        // We assume the caller passes the bottom-left coordinate of the trigger element
        const spaceBelow = viewportHeight - position.top
        const spaceAbove = position.top
        const popoverHeight = rect.height

        // If not enough space below and more space above, flip it
        if (spaceBelow < popoverHeight + 20 && spaceAbove > popoverHeight + 20) {
            newTop = position.top - popoverHeight - 10 // 10px buffer
        }

        // Horizontal clamping
        if (newLeft + rect.width > viewportWidth - 20) {
            newLeft = viewportWidth - rect.width - 20
        }
        if (newLeft < 20) {
            newLeft = 20
        }

        setAdjustedPosition({ top: newTop, left: newLeft })
    }, [position]) // Re-run when position changes

    return (
        <div
            ref={ref}
            style={{
                position: 'fixed',
                top: adjustedPosition.top,
                left: adjustedPosition.left,
                zIndex: 1000,
                width: '360px',
                maxWidth: '90vw',
                backgroundColor: '#18181b', // Zinc 900
                border: '1px solid #27272a',
                borderRadius: '12px',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.3), 0 2px 4px -2px rgb(0 0 0 / 0.3)',
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
                fontFamily: "'Inter', sans-serif"
            }}
        >
            <div
                style={{
                    padding: '8px 12px',
                    backgroundColor: '#09090b', // Zinc 950
                    borderBottom: '1px solid #27272a',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    fontSize: '12px',
                    fontWeight: 600,
                    color: '#e4e4e7',
                    height: '40px'
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ color: headerText === 'Bug' ? '#ef4444' : headerText === 'Investigate' ? '#f59e0b' : '#a1a1aa' }}>{headerText}</span>
                    <span style={{ color: '#71717a', fontSize: '11px', fontFamily: 'monospace' }}>{lineRange}</span>
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <button
                        onClick={onClose}
                        style={{
                            background: 'none',
                            border: 'none',
                            color: '#71717a',
                            cursor: 'pointer',
                            padding: '4px',
                            display: 'flex',
                        }}
                    >
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor">
                            <path d="M1 1L13 13M1 13L13 1" strokeWidth="1.5" strokeLinecap="round" />
                        </svg>
                    </button>
                </div>
            </div>

            <div
                style={{
                    padding: '16px',
                    fontSize: '13px',
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                    <div style={{ width: '16px', height: '16px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {/* Sparkle Icon */}
                        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ width: '100%', height: '100%', color: '#2dd4bf' }}>
                            <path d="M12 2L14.4 7.2L19.6 9.6L14.4 12L12 17.2L9.6 12L4.4 9.6L9.6 7.2L12 2Z" fill="currentColor" />
                        </svg>
                    </div>
                    <span style={{ fontWeight: 600, color: '#e4e4e7' }}>ASYNCREVIEW</span>
                </div>

                <div style={{ fontWeight: 500, color: '#e4e4e7', marginBottom: '8px' }}>
                    {issue.title}
                </div>

                <div style={{ color: '#a1a1aa', lineHeight: '1.6' }}>
                    <ReactMarkdown>{issue.explanationMarkdown}</ReactMarkdown>
                </div>
            </div>

            {(issue.fix_suggestions && issue.fix_suggestions.length > 0) && (
                <div style={{ padding: '0 16px 16px', fontSize: '12px' }}>
                    <div style={{ fontWeight: 600, color: '#e4e4e7', marginBottom: '8px' }}>Suggested Fix:</div>
                    <div style={{
                        backgroundColor: '#27272a',
                        padding: '8px',
                        borderRadius: '6px',
                        fontFamily: 'monospace',
                        color: '#e4e4e7',
                        whiteSpace: 'pre-wrap'
                    }}>
                        {issue.fix_suggestions[0]}
                    </div>
                </div>
            )}

            <div style={{
                padding: '8px 16px',
                borderTop: '1px solid #27272a',
                display: 'flex',
                gap: '8px',
                justifyContent: 'flex-end',
                backgroundColor: '#09090b'
            }}>
                <button style={{
                    background: '#27272a',
                    border: '1px solid #3f3f46',
                    borderRadius: '4px',
                    padding: '4px 8px',
                    fontSize: '11px',
                    color: '#e4e4e7',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                }}>
                    <span>✓</span> Mark resolved
                </button>
            </div>
        </div>
    )
}

export const IssuePopover = React.memo(IssuePopoverComponent)

